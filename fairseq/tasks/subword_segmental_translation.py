# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II

import numpy as np
from fairseq import metrics, utils, search
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

import re
from collections import Counter
import torch
import nltk
import copy

import time

EVAL_BLEU_ORDER = 4
ENCODING = "utf-8"
SPACE_NORMALIZER = re.compile(r"\s+")

logger = logging.getLogger(__name__)


def char_tokenize(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return list(line)


def tokenize_segs(line, max_seg_len, char_segs, non_alpha=False):
    # Split into all possible segments
    segs = []
    for n in range(1, max_seg_len+1):
        if n == 1 and not char_segs:
            continue

        chars = list(line)
        segs_n = nltk.ngrams(chars, n=n)
        segs_n = ["".join(seg) for seg in segs_n]

        if not non_alpha and n > 1:  # Discard segments with non-alphabetical characters
            segs_n = [seg for seg in segs_n if seg.isalpha() and len(seg) == n]
        else:
            segs_n = [seg for seg in segs_n if len(seg) == n]
        segs.extend(segs_n)
    return segs


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )

        # Read input sentences.
        tgt_path = data_path[0: data_path.rindex("/") + 1] + split + "." + tgt
        tgt_sentences, tgt_lengths = [], []
        with open(tgt_path, encoding=ENCODING) as file:
            for line in file:
                sentence = line.strip()

                # Tokenize the sentence, splitting on spaces
                tokens = tgt_dict.encode_line(
                    sentence, line_tokenizer=char_tokenize, add_if_not_exist=False,
                )

                tgt_sentences.append(tokens.type(torch.int64))
                tgt_lengths.append(tokens.numel())

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(tgt_sentences)
            )
        )

        if not combine:
            break

    return LanguagePairDataset(
        src=src_dataset,
        src_sizes=src_dataset.sizes,
        src_dict=src_dict,
        tgt=tgt_sentences,
        tgt_sizes=tgt_lengths,
        tgt_dict=tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=None,
        eos=None,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@dataclass
class SubwordSegmentalTranslationConfig(FairseqDataclass):
    max_seg_len: int = field(
        default=5,
        metadata={"help": "maximum segment length"},
    )
    lexicon_max_size: int = field(
        default=0,
        metadata={"help": "size of decoder subword lexicon"},
    )
    lexicon_min_count: int = field(
        default=1,
        metadata={"help": "minimum frequency for inclusion in lexicon"}
    )
    vocabs_path: Optional[str] = field(
        default=None,
        metadata={"help": "directory for storing and load lex and char vocabs"}
    )

    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
    )
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; thitgts is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # options for reporting BLEU during validation
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    # Generation params
    average_next_scores: bool = field(
        default=False, metadata={"help": "average next segment scores during decoding"}
    )
    normalize_type: Optional[str] = field(
        default=None, metadata={"help": "seg: /= # segs, char: /= (# segs + # chars)"}
    )
    decoding: Optional[str] = field(
        default="dynamic", metadata={"help": "decoding algorithm: dynamic, beam, separate"}
    )

    # Additional training params
    lex_only: bool = field(
        default=False, metadata={"help": "model segments using only the lexicon"}
    )


@register_task("subword_segmental_translation", dataclass=SubwordSegmentalTranslationConfig)
class SubwordSegmentalTranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: SubwordSegmentalTranslationConfig

    def __init__(self, cfg: SubwordSegmentalTranslationConfig, src_dict, tgt_dict, tgt_lex):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.tgt_lex = tgt_lex

    @classmethod
    def setup_task(cls, cfg: SubwordSegmentalTranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        vocab_path = cfg.vocabs_path
        try:
            logger.info("Trying to load existing target lexicon..")
            tgt_lex = cls.load_dictionary(
                os.path.join(vocab_path, "lex_dict.{}.txt".format(cfg.target_lang))
            )
            logger.info("Lexicon loaded.")
            logger.info("[{}] lexicon dictionary: {} types".format(cfg.target_lang, len(tgt_lex)))

        except FileNotFoundError:
            logger.info("Target lexicon dictionary file does not exist.")
            logger.info("Creating subword lexicon from word dictionary...")
            start_time = time.time()

            counter = Counter()
            symbols = []

            for index, word in enumerate(tgt_dict.symbols):
                if index < tgt_dict.nspecial:
                    counter[word] = tgt_dict.count[index]
                    symbols.append(word)
                else:
                    subwords = tokenize_segs(word, cfg.max_seg_len, char_segs=True, non_alpha=False)
                    subwords = {subword: tgt_dict.count[index] for subword in subwords} # Counter(subwords)
                    counter.update(subwords)
            print("--- Finished creating %s seconds ---" % (time.time() - start_time))

            nonspecial_symbols = [subword for subword in counter.keys() if subword not in symbols]
            symbols.extend(nonspecial_symbols)

            # Trim lexicon to maximum size
            assert cfg.lexicon_max_size > 0
            trimmed_counter = Counter()
            trimmed_symbols = []

            subword_counter = Counter()
            index = 0
            for subword in symbols:
                if index < tgt_dict.nspecial:
                    trimmed_counter[subword] = counter[subword]
                    trimmed_symbols.append(subword)
                else:
                    subword_counter[subword] = counter[subword]
                index += 1

            for subword, count in subword_counter.most_common(n=cfg.lexicon_max_size):
                trimmed_counter[subword] = count
                trimmed_symbols.append(subword)

            counter = trimmed_counter
            symbols = trimmed_symbols

            print("--- Finished trimming %s seconds ---" % (time.time() - start_time))

            # Add space to lexicon vocab
            space_word = " "
            counter[space_word] = 1
            symbols.append(space_word)

            tgt_lex = copy.deepcopy(tgt_dict)
            tgt_lex.indices = {subword: index for index, subword in enumerate(symbols)}
            tgt_lex.count = [counter[subword] for subword in symbols]
            tgt_lex.symbols = symbols

            tgt_lex.save(os.path.join(vocab_path, "lex_dict.{}.txt".format(cfg.target_lang)))
            logger.info("Target lexicon dictionary saved to file: ")
            logger.info(os.path.join(vocab_path, "lex_dict.{}.txt".format(cfg.target_lang)))

            print("--- Finished saving %s seconds ---" % (time.time() - start_time))

        # START
        try:
            logger.info("Trying to load existing target char vocab..")
            tgt_dict = cls.load_dictionary(
                os.path.join(vocab_path, "char_dict.{}.txt".format(cfg.target_lang))
            )
            logger.info("Char vocab loaded.")
            logger.info("[{}] char dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

            # Add end-of-morpheme token to char vocab
            eom_word = "<eom>"
            tgt_dict.symbols = tgt_dict.symbols[0: tgt_dict.nspecial] + [eom_word] + tgt_dict.symbols[tgt_dict.nspecial:]
            tgt_dict.count = tgt_dict.count[0: tgt_dict.nspecial] + [1] + tgt_dict.count[tgt_dict.nspecial:]
            tgt_dict.indices = {char: index for index, char in enumerate(tgt_dict.symbols)}
            tgt_dict.nspecial += 1
        except FileNotFoundError:
            logger.info("Target char dictionary file does not exist.")
            #END and untab from here

            logger.info("Creating char vocab from word dictionary...")
            # CREATE CHARACTER TGT_DICT
            counter = Counter()
            symbols = []
            for index, word in enumerate(tgt_dict.symbols):
                if index < tgt_dict.nspecial:
                    counter[word] = tgt_dict.count[index]
                    symbols.append(word)
                else:
                    chars = {char: word.count(char) for char in word}
                    counter.update(chars)

            nonspecial_symbols = [char for char in counter.keys() if char not in symbols]
            symbols.extend(nonspecial_symbols)

            # Add space to char vocab
            space_word = " "
            counter[space_word] = 1
            symbols.append(space_word)

            # Add end-of-morpheme token to char vocab
            eom_word = "<eom>"
            counter[eom_word] = 1
            symbols = symbols[0: tgt_dict.nspecial] + [eom_word] + symbols[tgt_dict.nspecial: ]
            tgt_dict.nspecial += 1

            tgt_dict.indices = {char: index for index, char in enumerate(symbols)}
            tgt_dict.count = [counter[char] for char in symbols]
            tgt_dict.symbols = symbols

            # Sort tgt_dict special characters, then alphabetical characters, then the rest
            special_tokens = []
            alpha_chars = []
            non_alpha_chars = []
            for index, char in enumerate(tgt_dict.symbols):
                if index < tgt_dict.nspecial:
                    special_tokens.append(char)
                elif char.isalpha():
                    alpha_chars.append(char)
                else:
                    non_alpha_chars.append(char)

            tgt_dict.symbols = special_tokens + alpha_chars + non_alpha_chars
            for index, char in enumerate(tgt_dict.symbols):
                tgt_dict.indices[char] = index

            tgt_dict.save(os.path.join(vocab_path, "char_dict.{}.txt".format(cfg.target_lang)))
            logger.info("Target char dictionary saved to file:" )
            logger.info(os.path.join(vocab_path, "char_dict.{}.txt".format(cfg.target_lang)))

        return cls(cfg, src_dict, tgt_dict, tgt_lex)

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        """
        Build a :class:`~fairseq.SequenceGenerator` instance for this
        task.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            args (fairseq.dataclass.configs.GenerationConfig):
                configuration object (dataclass) for generation
            extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
                through to SequenceGenerator
            prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
                If provided, this function constrains the beam search to
                allowed tokens only at each step. The provided function
                should take 2 arguments: the batch ID (`batch_id: int`)
                and a unidimensional tensor of token ids (`inputs_ids:
                torch.Tensor`). It has to return a `List[int]` with the
                allowed tokens for the next generation step conditioned
                on the previously generated tokens (`inputs_ids`) and
                the batch ID (`batch_id`). This argument is useful for
                constrained generation conditioned on the prefix, as
                described in "Autoregressive Entity Retrieval"
                (https://arxiv.org/abs/2010.00904) and
                https://github.com/facebookresearch/GENRE.
        """
        args, task_args = args  # seperate default generation args from task-specific args

        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.subword_segmental_generator import (
            SubwordSegmentalGenerator
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        decoding = getattr(task_args, "decoding")
        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            seq_gen_cls = SubwordSegmentalGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            average_next_scores=getattr(task_args, "average_next_scores", False),
            normalize_type=getattr(task_args, "normalize_type", None),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]

        # Print example segmentations
        # sample["net_input"]["mode"] = "segment"
        # Subset of validation set
        # n = 1
        # sample["id"] = sample["id"][0: n]
        # sample["nsentences"] = n
        #
        # sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"][0: n]
        # sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"][0: n]
        # sample["net_input"]["prev_output_tokens"] = sample["net_input"]["prev_output_tokens"][0: n]
        # sample["target"] = sample["target"][0: n]
        # sample["ntokens"] = torch.numel(sample["target"])

        # split_indices = self.segment(sample, model, criterion)
        # split_text = self.split_text(sample, split_indices)
        # print(split_text)

        return loss, sample_size, logging_output

    def split_text(self, sample, split_indices, num_examples=10):
        target_ids = sample["target"].transpose(0, 1)

        batch_size = target_ids.shape[1]
        seq_len = target_ids.shape[0]
        batch_texts = []
        num_examples = min(batch_size, num_examples)

        eos_ends = []

        for i in range(num_examples):
            eos_ends.append(False)
            seq_text = ""
            for j in range(seq_len):
                if target_ids[j][i] == self.tgt_dict.eos_index:
                    eos_ends[i] = True
                    break
                seq_text += self.tgt_dict.symbols[target_ids[j][i]]

            seq_text = seq_text.replace("</s>", " ")
            batch_texts.append(seq_text)

        split_texts = []
        for i, text in enumerate(batch_texts):
            for counter, index in enumerate(split_indices[i]):
                text = text[:index + counter + 1] + "|" + text[index + counter + 1:]

            if eos_ends[i]:
                text = text[0: -1] + "</s>"

            split_texts.append(text)

        return split_texts

    def segment(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            split_indices = criterion(model, sample)
        return split_indices

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect

                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    @property
    def target_lexicon(self):
        """Return the target lexicon :class:`~fairseq.data.Dictionary`."""
        return self.tgt_lex

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
