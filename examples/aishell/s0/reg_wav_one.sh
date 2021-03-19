#!/bin/bash
. path.sh
wav=$1

dict=data/dict/lang_char.txt
model_dir=exp/train_unified_conformer/
model_path=$model_dir/final.pt

python wenet/bin/recognize_pinyin_one.py --gpu -1 \
    --mode attention_rescoring \
    --config $model_dir/train.yaml \
    --test_data $wav \
    --checkpoint $model_path \
    --beam_size 3 \
    --batch_size 1 \
    --penalty 0.0 \
    --dict $dict \
    --result_file text_result
