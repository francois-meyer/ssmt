# SSMT

Code for the paper *Subword Segmental Machine Translation: Unifying Segmentation and Target Sentence Generation*, Francois Meyer and Jan Buys, Findings of ACL 2023.

SSMT is implemented as a model in fairseq. The code in this repo can be used to train new SSMT models and to generate translations with trained models using dynamic decoding. The SSMT models trained for our paper (English to Xhosa, Zulu, Swati, Finnish, Afrikaans, and Tswana) are publicly available:
* [Trained SSMT models](https://drive.google.com/file/d/1ElJCkpyjnZt7ry--cOqZRXgWWCrefHoh/view?usp=share_link)

## Dependencies
* python 3
* [fairseq](https://github.com/pytorch/fairseq) (commit: 806855bf660ea748ed7ffb42fe8dcc881ca3aca0)
* pytorch 1.0.1.post2
* cuda 11.4
* nltk

## Usage
Merge the ssmt files with fairseq.

```shell
git clone https://github.com/pytorch/fairseq.git
git clone https://github.com/francois-meyer/ssmt

# change to 806855bf660ea748ed7ffb42fe8dcc881ca3aca0 branch
cd fairseq
git checkout 806855bf660ea748ed7ffb42fe8dcc881ca3aca0 

# copy files from ssmt to fairseq
cp -r ../ssmt/fairseq ./ 
cp -r ../ssmt/fairseq_cli ./  
```

## Instructions

1. Segment source language dataset with BPE, leave target language dataset unsegmented.

2. Run fairseq-preprocess to build vocabularies and binarise data.

3. Train SSMT model with fairseq.

```shell
python fairseq/fairseq_cli/train.py \
    $DATA_DIR --task subword_segmental_translation --source-lang eng --target-lang $LANG\
    --max-epoch 30 --arch ssmt --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion subword_segmental_cross_entropy --label-smoothing 0.1 \
    --max-tokens 12288 --vocabs-path $OUT_DIR \
    --no-epoch-checkpoints --keep-best-checkpoints 1  \
    --max-seg-len 5 --lexicon-max-size 5000 \
    --ddp-backend=pytorch_ddp --num-workers 0 \
    --save-dir $OUT_DIR &>> $OUT_DIR/log
```

4. Run generate_ssmt.py to generate translations.

```shell
python fairseq/fairseq_cli/generate_ssmt.py $DATA_DIR \
    --task subword_segmental_translation --source-lang eng --target-lang $LANG \
    --path $MODEL_DIR/checkpoint_best.pt --max-len-b 500 \
    --vocabs-path $OUT_DIR \
    --normalize-type seg-seg  \
    --max-seg-len 5 --lexicon-max-size 5000 --batch-size 4 --beam 5 --results-path $RESULTS_DIR &>> $RESULTS_DIR/log
```
