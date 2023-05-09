# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.ssmt import SubwordSegmentalConfig
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch.nn.functional as F
LOGINF = 1000000.0


def map_chars2lex(tgt_dict, tgt_lex):
    chars2lex = {}
    for lex_id, subword in enumerate(tgt_lex.symbols):
        if lex_id < tgt_lex.nspecial:
            special_char_id = (tgt_dict.indices[subword], )
            chars2lex[special_char_id] = lex_id
        else:
            seg_chars = tuple(subword)
            seg_char_ids = tuple((tgt_dict.indices[char] for char in seg_chars))
            chars2lex[seg_char_ids] = lex_id
    return chars2lex


def map_prechars2lex(chars2lex):
    char_indices = list(chars2lex.keys())
    prechars2lex = {}
    for chars in char_indices:
        seg_len = len(chars)
        if seg_len == 1:
            continue
        for end in range(1, seg_len):
            prechars = chars[0: end]
            if prechars in prechars2lex:
                prechars2lex[prechars].append(chars2lex[chars])
            else:
                prechars2lex[prechars] = [chars2lex[chars]]
    return prechars2lex


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerDecoderBase":
        return "TransformerDecoder"
    else:
        return module_name


class SubwordSegmentalDecoderBase(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        lexicon,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
        max_seg_len=None
    ):

        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = cfg.decoder.layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder.embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.decoder.output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(cfg, no_encoder_attn)
                for _ in range(cfg.decoder.layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.decoder.normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)

        # New class variables
        self.max_seg_len = max_seg_len
        self.char_decoder = LSTMCharDecoder(
            vocab_size=len(dictionary.symbols),
            input_size=input_embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            dropout=cfg.dropout
        )

        eom_word = "<eom>"
        self.eom_id = dictionary.indices[eom_word]
        self.reg_exp = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.char_nspecial = dictionary.nspecial
        self.char_nalpha = len([char for char in dictionary.symbols if char.isalpha()])
        self.chars2lex = map_chars2lex(tgt_dict=dictionary, tgt_lex=lexicon)
        self.prechars2lex = map_prechars2lex(self.chars2lex)
        self.lex_vocab_size = len(lexicon.symbols)
        self.lex_decoder = LexDecoder(
            vocab_size=self.lex_vocab_size,
            hidden_size=embed_dim,
            num_layers=1,
            dropout=cfg.dropout
        )

        self.mixture_gate_func = nn.Linear(embed_dim, 1)
        self.generate = False

    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if cfg.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5
            )
        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = transformer_layer.TransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        mode = "forward"
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        if self.generate:
            if len(prev_output_tokens) == 2:
                beam_ids, prev_seg_ends = prev_output_tokens
                prev_history_encodings = None
            else:
                beam_ids, prev_seg_ends, prev_history_encodings = prev_output_tokens
            beam_ids = beam_ids.to(self.device)

            x, positioned_input_embeddings, extra = self.extract_features(
                beam_ids,  # (batch_size, tgt_cur_len)
                encoder_out=encoder_out,  # (src_seq_len, batch_size, embed_dim)
                incremental_state=incremental_state,
                full_context_alignment=full_context_alignment,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
            )

            history_encodings = x  # (batch_size, tgt_cur_len, embed_dim)
            if prev_history_encodings is not None:
                history_encodings = torch.cat([prev_history_encodings, history_encodings], dim=1)

            char_vocab_lprobs, char_eom_lprobs = self.inference(
                beam_ids,
                history_encodings,
                prev_seg_ends
            )
            return char_vocab_lprobs, char_eom_lprobs, history_encodings
        else:
            # Encode character-level histories
            x, positioned_input_embeddings, extra = self.extract_features(
                prev_output_tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
                full_context_alignment=full_context_alignment,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
            )

            # Prepare ids and history embeddings from Transformer encoder
            input_ids = torch.transpose(prev_output_tokens, 0, 1)  # [seq_len + 1, batch_size]
            target_ids = input_ids[1:, :]  # [seq_len, batch_size]
            input_embeddings = self.get_input_embeddings(prev_output_tokens)
            input_embeddings = torch.transpose(input_embeddings, 0, 1)  # [seq_len + 1, batch_size, embed_dim]
            history_encodings = torch.transpose(x, 0, 1)[0: -1]  # [seq_len, batch_size, embed_dim]
            batch_size = target_ids.shape[1]
            seq_len = target_ids.shape[0]

            # Collect segment input embeddings and target ids
            seg_embeddings, seg_target_ids, seg_lens, seq_lens = self.collect_segs(
                target_ids,
                input_embeddings,
                batch_size,
                seq_len
            )

            # Compute character-by-character probabilities
            char_logp, full_char_logp = self.compute_char_lprobs(
                history_encodings,
                seg_embeddings,
                seg_target_ids
            )

            # Compute lexicon probabilities
            if self.lex_vocab_size > 0:
                lex_logp = self.compute_lex_lprobs(
                    history_encodings,
                    target_ids,
                    seq_len
                )

                # Compute mixture coefficent
                log_char_proportions, log_lex_proportions = self.compute_mix_lcoefs(
                    history_encodings
                )

                # Compute segment probabilities
                seg_logp = self.compute_seg_lprobs(
                    char_logp,
                    full_char_logp,
                    seg_lens,
                    batch_size,
                    seq_len,
                    lex_logp,
                    log_char_proportions,
                    log_lex_proportions
                )

            else:
                # Compute segment probabilities
                seg_logp = self.compute_seg_lprobs(
                    char_logp,
                    full_char_logp,
                    seg_lens,
                    batch_size,
                    seq_len
                )

            # Dynamic programming
            if mode == "forward":
                # Compute marginals
                log_alpha, log_R = self.forward_pass(
                    seg_logp,
                    batch_size,
                    seq_len
                )
                return log_alpha, log_R, seq_lens
            elif mode == "segment":
                # Compute maximising segmentation
                split_indices = self.viterbi(
                    seg_logp,
                    seq_lens
                )
                return split_indices

    def get_input_embeddings(self, input_ids):
        embeddings = self.embed_tokens(input_ids)
        embeddings = self.embed_scale * embeddings
        embeddings = self.dropout_module(embeddings)
        return embeddings

    def inference(self,
        beam_ids,  # (batch_size * beam_size, tgt_cur_len)
        history_encodings,   # (batch_size * beam_size, tgt_cur_len, embed_dim)
        prev_seg_ends  # (batch_size * beam_size)
    ):

        num_beams = beam_ids.shape[0]
        seq_len = beam_ids.shape[1]
        cur_seg_lens = [seq_len - prev_seg_end for prev_seg_end in prev_seg_ends]
        prev_seg_ends = torch.tensor(prev_seg_ends).to(self.device)
        cur_seg_lens = torch.tensor(cur_seg_lens).to(self.device)
        char_vocab_size = self.char_decoder.vocab_size

        # Gather relevant history encoding for each sequence
        history_indices = prev_seg_ends.unsqueeze(-1)
        history_indices = history_indices.repeat(1, history_encodings.shape[-1])
        history_indices = history_indices.unsqueeze(1)
        history_encoding = torch.gather(history_encodings, dim=1, index=history_indices)
        history_encoding = history_encoding.squeeze(1)  # (batch_beam_size, embed_dim)

        lex_con_lprobs, lex_eom_lprobs = self.lex_inference(beam_ids, history_encoding, prev_seg_ends, cur_seg_lens,
                                                             char_vocab_size)

        # Extract history encoding up to last segment, prepare for all possible segments
        history_encoding = history_encoding.unsqueeze(0)  # (1, num_beams, embed_dim)
        history_encoding = torch.repeat_interleave(history_encoding, char_vocab_size, dim=1)  # (1, num_beams * char_vocab_size, embed_dim)

        # Collect and pad segment ids
        char_vocab_ids = torch.arange(char_vocab_size).to(self.device).unsqueeze(1)
        seg_ids = []
        seg_lens = []
        batch_lens = []
        for beam_num in range(num_beams):
            beam_seg_ids = beam_ids[beam_num, prev_seg_ends[beam_num]: ]
            seg_len = len(beam_seg_ids) + 1
            batch_lens.append(seg_len)
            seg_lens.extend([seg_len] * char_vocab_size)
            beam_seg_ids = beam_seg_ids.repeat(char_vocab_size, 1)
            beam_seg_ids = torch.cat([beam_seg_ids, char_vocab_ids], dim=1).transpose(0, 1)
            seg_ids.append(beam_seg_ids)
        padded_seg_ids = pad_sequence(seg_ids, padding_value=self.padding_idx)
        padded_seg_ids = padded_seg_ids.view(padded_seg_ids.shape[0], -1)

        # Extract input embeddings for current segments
        padded_seg_embeddings = self.get_input_embeddings(padded_seg_ids)  # (max_seg_len, batch_size * beam_size, embed_dim)
        packed_seg_embeddings = pack_padded_sequence(padded_seg_embeddings, torch.tensor(seg_lens), enforce_sorted=False)

        # Produce probabilities for segments including next characters and next characters+eom
        char_logits, _ = self.char_decoder(packed_seg_embeddings, history_encoding)
        char_lprobs = self.get_normalized_probs(net_output=(char_logits,), log_probs=True)  # (cur_seg_len+1, char_vocab_size, char_vocab_size)

        # Collect char probabilities
        char_vocab_lprobs = []
        eom_lprobs = []
        for beam_num in range(num_beams):
            beam_char_lprobs = char_lprobs[0: cur_seg_lens[beam_num] + 1,
                                            beam_num * char_vocab_size: beam_num * char_vocab_size + char_vocab_size, :]
            prev_char_lprob = 0.0
            if cur_seg_lens[beam_num] > 1:
                prev_char_lprobs = torch.gather(beam_char_lprobs[0: -2, 0], dim=-1,
                                                index=beam_ids[beam_num, prev_seg_ends[beam_num]+1:].unsqueeze(-1))
                prev_char_lprob = torch.sum(prev_char_lprobs)
            beam_char_vocab_lprobs = prev_char_lprob + beam_char_lprobs[-2, 0, :]  # add second last output to chain rule, for next character
            beam_eom_lprobs = beam_char_lprobs[-1, :, self.eom_id]  # last output for eom
            char_vocab_lprobs.append(beam_char_vocab_lprobs)
            eom_lprobs.append(beam_eom_lprobs)

        char_vocab_lprobs = torch.cat(char_vocab_lprobs, dim=0).view(num_beams, -1)
        eom_lprobs = torch.cat(eom_lprobs, dim=0).view(num_beams, -1)

        char_eom_lprobs = char_vocab_lprobs + eom_lprobs

        # Next character cannot be <pad>
        char_eom_lprobs[:, self.padding_idx] = -LOGINF
        char_vocab_lprobs[:, self.padding_idx] = -LOGINF

        # Next character cannot be eom itself
        char_eom_lprobs[:, self.eom_id] = -LOGINF
        char_vocab_lprobs[:, self.eom_id] = -LOGINF

        # Non-alphabetic character can only be one-character segment
        char_vocab_lprobs[:, 0: self.char_nspecial] = -LOGINF
        char_vocab_lprobs[:, self.char_nspecial + self.char_nalpha: ] = -LOGINF

        for beam_num in range(num_beams):
            # Only alphabetic characters can be part of multi-character segment
            if cur_seg_lens[beam_num] > 1:
                char_eom_lprobs[beam_num, 0: self.char_nspecial] = -LOGINF
                char_eom_lprobs[beam_num,self.char_nspecial + self.char_nalpha:] = -LOGINF

            # Segment cannot continue further if next character alread makes it as long as allowed
            if cur_seg_lens[beam_num] >= self.max_seg_len:
                char_vocab_lprobs[beam_num, :] = -LOGINF

        history_encoding = torch.gather(history_encodings, dim=1, index=history_indices)
        log_char_proportions, log_lex_proportions = self.compute_mix_lcoefs(
            history_encoding
        )

        char_con_element = log_char_proportions + char_vocab_lprobs
        lex_con_element = log_lex_proportions + lex_con_lprobs
        con_lprobs = torch.logsumexp(torch.stack([char_con_element, lex_con_element]), dim=0)

        char_eom_element = log_char_proportions + char_eom_lprobs
        lex_eom_element = log_lex_proportions + lex_eom_lprobs
        eom_lprobs = torch.logsumexp(torch.stack([char_eom_element, lex_eom_element]), dim=0)

        return con_lprobs, eom_lprobs

    def lex_inference(self,
        beam_ids,  # (batch_size * beam_size, tgt_cur_len)
        history_encoding,  # (batch_size * beam_size, tgt_cur_len, embed_dim)
        prev_seg_ends,  # (batch_size * beam_size)
        cur_seg_lens,   # (batch_size * beam_size)
        char_vocab_size
    ):

        num_beams = beam_ids.shape[0]

        # Compute lexicon lprobs
        lex_logits = self.lex_decoder(history_encoding)  # (batch_size * beam_size, lex_vocab_size)
        full_lex_logp = self.get_normalized_probs(net_output=(lex_logits,),
                                                  log_probs=True)  # in models.fairseq_decoder  # (lex_vocab_size)
        full_lex_logp = torch.cat((full_lex_logp, torch.full((full_lex_logp.shape[0], 1), fill_value=-LOGINF,
                                                             device=self.device)), dim=-1)

        # Prepare indices
        target_ids = beam_ids.transpose(0, 1)  # (tgt_cur_len, batch_size * beam_size)
        target_ids = torch.repeat_interleave(target_ids, char_vocab_size, dim=1)  # (tgt_cur_len, batch_size * beam_size * char_vocab_size)
        char_vocab_ids = torch.arange(char_vocab_size).repeat(num_beams).unsqueeze(0).to(self.device)
        target_ids = torch.cat([target_ids, char_vocab_ids], dim=0)
        seg_starts = [prev_seg_end + 1 for prev_seg_end in prev_seg_ends]

        seg_char_ids = []
        for beam_num in range(num_beams):
            beam_seg_char_ids = target_ids[seg_starts[beam_num]: seg_starts[beam_num] + cur_seg_lens[beam_num],
                                    beam_num * char_vocab_size: beam_num * char_vocab_size + char_vocab_size].T.tolist()
            beam_seg_char_ids = [tuple(char_ids) for char_ids in beam_seg_char_ids]
            seg_char_ids.extend(beam_seg_char_ids)

        # Collect ended segment lprobs
        seg_lex_ids = torch.LongTensor([self.chars2lex[char_ids] if char_ids in self.chars2lex
                                        else self.lex_vocab_size  # not in segment lexicon
                                        for char_ids in seg_char_ids]).to(self.device)
        seg_lex_ids = seg_lex_ids.view(num_beams, -1)
        lex_eom_logp = torch.gather(full_lex_logp, dim=-1, index=seg_lex_ids).squeeze(-1)

        # Collect continued segment lprobs
        preseg_lex_ids = [self.prechars2lex[char_ids] if char_ids in self.prechars2lex
                          else [self.lex_vocab_size]  # not in pre-segment lexicon
                          for char_ids in seg_char_ids]
        max_lex_mappings = max([len(lex_ids) for lex_ids in preseg_lex_ids])
        for i in range(len(preseg_lex_ids)):
            num_lex_mappings = len(preseg_lex_ids[i])
            preseg_lex_ids[i].extend([self.lex_vocab_size] * (max_lex_mappings - num_lex_mappings))
        preseg_lex_ids = torch.tensor(preseg_lex_ids).view(num_beams, char_vocab_size * max_lex_mappings).to(self.device)
        lex_con_logp = torch.gather(full_lex_logp, dim=-1, index=preseg_lex_ids).view(num_beams, char_vocab_size, max_lex_mappings)
        lex_con_logp = torch.logsumexp(lex_con_logp, dim=-1)

        return lex_con_logp, lex_eom_logp

    def collect_segs(
        self,
        target_ids,
        input_embeddings,
        batch_size,
        seq_len
    ):

        alpha_mask = (target_ids >= self.char_nspecial) & (target_ids < self.char_nspecial + self.char_nalpha)
        target_alphabet = torch.where(alpha_mask, True, False)
        pad_mask = torch.where(target_ids != self.padding_idx, True, False)
        seq_lens = torch.sum(pad_mask, dim=0)
        seg_embeddings = []
        seg_target_ids = []
        seg_lens = []

        for seg_end in range(self.max_seg_len - 1,
                             seq_len + self.max_seg_len - 1):  # end of first possible seg to end of sequence

            seg_target_alphas = target_alphabet[seg_end - self.max_seg_len + 1: seg_end + 1]
            seg_lens.append([])
            for seq_num in range(batch_size):
                if not seg_target_alphas[0][seq_num]:
                    seg_len = 1
                else:
                    for j in range(len(seg_target_alphas)):
                        if seg_target_alphas[j][seq_num]:
                            seg_len = j + 1
                        else:
                            break
                seg_lens[-1].append(seg_len)

            seg_embeddings.append(
                input_embeddings[seg_end - self.max_seg_len + 1: seg_end + 2].clone())
            seg_target_ids.append(target_ids[seg_end - self.max_seg_len + 1: seg_end + 1].clone())

            if seg_end >= seq_len:  # pad with zero embeddings and pad ids
                seg_embeddings[-1] = torch.cat((seg_embeddings[-1], torch.zeros((seg_end - seq_len + 1, batch_size,
                                                                                 input_embeddings.shape[-1]),
                                                                                 device=self.device)))
                                                                                # device=self.device)))
                seg_target_ids[-1] = torch.cat((seg_target_ids[-1], torch.full((seg_end - seq_len + 1, batch_size),
                                                                               fill_value=self.padding_idx,
                                                                               device=self.device)))
                                                                               # device=self.device)))

        seg_embeddings = torch.stack(seg_embeddings, dim=1).view(self.max_seg_len + 1, -1, input_embeddings.shape[2])
        return seg_embeddings, seg_target_ids, seg_lens, seq_lens

    def compute_lex_lprobs(
        self,
        history_encodings,  # (seq_len, batch_size, embed_dim)
        target_ids,  # (seq_len, batch_size)
        seq_len
    ):
        # Lexicon generation
        lex_logits = self.lex_decoder(history_encodings)
        full_lex_logp = self.get_normalized_probs(net_output=(lex_logits,),
                                                   log_probs=True)  # in models.fairseq_decoder
        full_lex_logp = torch.cat((full_lex_logp, torch.full((full_lex_logp.shape[0], full_lex_logp.shape[1], 1),
                                                             fill_value=-LOGINF, device=self.device)), dim=-1)

        lex_logp = {}
        for seg_len in range(1, self.max_seg_len + 1):
            lex_logp[seg_len] = []

            for seg_start in range(seq_len - (seg_len - 1)):
                seg_char_ids = target_ids[seg_start: seg_start + seg_len].T.tolist()
                seg_char_ids = [tuple(char_ids) for char_ids in seg_char_ids]
                seg_lex_ids = torch.LongTensor([self.chars2lex[char_ids] if char_ids in self.chars2lex
                                                else self.lex_vocab_size  # not in segment lexicon
                                                for char_ids in seg_char_ids]).to(self.device)
                lex_logp[seg_len].append(torch.gather(full_lex_logp[seg_start], dim=-1,
                                                      index=seg_lex_ids.unsqueeze(-1)).squeeze(-1))
            lex_logp[seg_len] = torch.stack(lex_logp[seg_len], dim=0)

        return lex_logp

    def compute_char_lprobs(
        self,
        history_encodings,
        seg_embeddings,
        seg_target_ids
    ):
        seg_hidden_states = history_encodings.contiguous().view(1, -1, history_encodings.shape[2])
        char_logits, _ = self.char_decoder(seg_embeddings, seg_hidden_states)
        full_char_logp = self.get_normalized_probs(net_output=(char_logits,),
                                                   log_probs=True)  # in models.fairseq_decoder
        target_prob_ids = torch.stack(seg_target_ids, dim=1).view(self.max_seg_len, -1)
        char_logp = torch.gather(
            full_char_logp[0: self.max_seg_len],
            dim=-1,
            index=target_prob_ids.unsqueeze(-1)
        ).squeeze(-1)

        return char_logp, full_char_logp

    def compute_mix_lcoefs(self,
        history_encodings
    ):
        logits = self.mixture_gate_func(history_encodings).squeeze(-1)

        log_char_proportions = F.logsigmoid(logits)
        log_lex_proportions = F.logsigmoid(-logits)

        return log_char_proportions, log_lex_proportions

    def compute_seg_lprobs(
        self,
        char_logp,
        full_char_logp,
        seg_lens,
        batch_size,
        seq_len,
        lex_logp=None,
        log_char_proportions=None,
        log_lex_proportions=None,
    ):
        seg_logp = {}
        for seg_len in range(1, self.max_seg_len + 1):
            end_batch_index = (seq_len - (seg_len - 1)) * batch_size

            seg_logp[seg_len] = torch.sum(char_logp[0: seg_len, 0: end_batch_index], dim=0) \
                                + full_char_logp[seg_len, 0: end_batch_index, self.eom_id]
            seg_logp[seg_len] = seg_logp[seg_len].view(-1, batch_size)

            # Keep only valid subwords
            valid_segs = torch.tensor(seg_lens[0: seq_len - (seg_len - 1)], device=self.device) >= seg_len
            seg_logp[seg_len] = torch.where(valid_segs, seg_logp[seg_len], torch.full_like(seg_logp[seg_len],
                                                                                               fill_value=-LOGINF))

            if self.lex_vocab_size > 0:
                # Calculate weighted average of character and lexical generation probabilities
                seg_log_char_proportions = log_char_proportions[0: seq_len - (seg_len - 1)]
                seg_log_lex_proportions = log_lex_proportions[0: seq_len - (seg_len - 1)]

                neginf_log_proportions = torch.full_like(seg_log_lex_proportions, fill_value=-LOGINF,
                                                         device=self.device)
                seg_log_lex_proportions = torch.where(lex_logp[seg_len] > -LOGINF, seg_log_lex_proportions,
                                                      neginf_log_proportions)

                char_element = seg_log_char_proportions + seg_logp[seg_len]
                lex_element = seg_log_lex_proportions + lex_logp[seg_len]
                seg_logp[seg_len] = torch.logsumexp(torch.stack([char_element, lex_element]), dim=0)

        return seg_logp

    def forward_pass(self,
        seg_logp,
        batch_size,
        seq_len
    ):
        # Compute alpha values and expected length factor
        log_alpha = torch.zeros((seq_len + 1, batch_size), device=self.device)
        if self.reg_exp > 1:
            log_pv = torch.zeros((seq_len + 1, batch_size), device=self.device)

        for t in range(1, seq_len + 1):  # from alpha_1 to alpha_bptt_len

            range_j = list(range(max(0, t - self.max_seg_len), t))

            log_alphas_t = log_alpha[range_j[0]: range_j[-1] + 1]
            seg_logp_elements = []
            regs_t = torch.zeros((len(range_j), 1)).to(self.device)

            for j in range_j:
                seg_logp_elements.append(seg_logp[t - j][j])
                regs_t[j - range_j[0]] = torch.log(torch.FloatTensor([(t - j) ** self.reg_exp]))

            seg_logp_t = torch.stack(seg_logp_elements)
            log_alpha[t] = torch.logsumexp(log_alphas_t + seg_logp_t, dim=0)

            if self.reg_exp > 1:
                log_p1v2_t = log_alphas_t + seg_logp_t + regs_t

                mask = torch.zeros_like(seg_logp_t)
                mask[0] = torch.tensor(-LOGINF)
                log_p2v1_t = seg_logp_t + log_pv[range_j[0]: range_j[-1] + 1] + mask

                pv_sum_elements = torch.logsumexp(torch.stack([log_p1v2_t, log_p2v1_t]), dim=0)
                log_pv[t] = torch.logsumexp(pv_sum_elements, dim=0)

        if self.reg_exp > 1:
            log_R = log_pv[-1] - log_alpha[-1]
        else:
            log_R = -LOGINF

        return log_alpha, log_R

    def viterbi(self,
        seg_logp,
        seq_lens
    ):
        split_indices = []
        for seq_num, seq_len in enumerate(seq_lens):
            # Compute alpha values and store backpointers
            bps = torch.zeros((self.max_seg_len, seq_len + 1), dtype=torch.long, device=self.device)
            max_logps = torch.full((self.max_seg_len, seq_len + 1), fill_value=0.0, device=self.device)
            for t in range(1, seq_len + 1):  # from alpha_1 to alpha_bptt_len
                for j in range(max(0, t - self.max_seg_len), t):
                    # The current segment starts at j and ends at t
                    # The backpointer will point to the segment ending at j-1
                    max_bp = max(1, j)  # Maximum possible length of segment ending at j-1

                    # Compute the probability of the most likely sequence ending with the segment j-t (length t-j)
                    # For this most likely sequence ending at j-1, what is the final segment length?
                    bps[t - j - 1, t] = torch.argmax(max_logps[0: max_bp, j])

                    # What is the probability of the most likely sequence ending at t with last segment j-t?
                    max_logps[t - j - 1, t] = torch.max(max_logps[0: max_bp, j]) + seg_logp[t - j][j, seq_num]

            # Backtrack from final state of most likely path
            best_path = []
            k = torch.tensor(seq_len)
            bp = torch.argmax(max_logps[:, seq_len])

            while k > 0:
                best_path.insert(0, torch.tensor(k) - 1)
                prev_bp = bp
                bp = bps[bp, k]
                k = k - (prev_bp + 1)
            split_indices.append(best_path)

        return split_indices


    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        input_embeddings = x.clone().detach()

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, input_embeddings, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class LexDecoder(nn.Module):
    """
    Once-off lexical generation of a segment, conditioned on the sequence history.
    """
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super(LexDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.drop = nn.Dropout(dropout)  # dropout used for embedding and final layer
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.drop(hidden_states)
        logits = self.fc(hidden_states)
        return logits


class LSTMCharDecoder(nn.Module):
    """
    Character by character generation of a segment, conditioned on the sequence history.
    """
    def __init__(self, vocab_size, input_size, hidden_size, num_layers, dropout):
        super(LSTMCharDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.drop = nn.Dropout(dropout)  # dropout used for embedding and final layer
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, embedding, init_hidden_states):
        """
        :param embedding: (seg_len + 1, num_segs, embedding_dim)
        :param init_hidden_states: (1, num_segs, embedding_dim)
        :return: logits: (seg_len + 1, num_segs, vocab_size)
        """
        init_hidden_states = self.transform(init_hidden_states)
        init_cell_states = torch.zeros_like(init_hidden_states)
        hidden_states, final_states = self.lstm(embedding, (init_hidden_states, init_cell_states))
        if type(hidden_states) is PackedSequence:
            hidden_states, _ = pad_packed_sequence(hidden_states)
        output = self.drop(hidden_states)
        logits = self.fc(output)
        return logits, (final_states[0].detach(), final_states[1].detach())

    def init_states(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class SubwordSegmentalDecoder(SubwordSegmentalDecoderBase):
    def __init__(
        self,
        args,
        dictionary,
        lexicon,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            SubwordSegmentalConfig.from_namespace(args),
            dictionary,
            lexicon,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(
            SubwordSegmentalConfig.from_namespace(args), dictionary, embed_tokens
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return super().build_decoder_layer(
            SubwordSegmentalConfig.from_namespace(args), no_encoder_attn=no_encoder_attn
        )