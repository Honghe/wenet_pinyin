#!/bin/bash
. path.sh
wav=$1
echo -e "utt:1\tfeat:$wav\tfeat_shape:5\ttext:一\ttoken:一\ttokenid:1\ttoken_shape:1,1" > format.data.tmp

dict=data/dict/lang_char.txt
model_dir=exp/train_unified_conformer/
model_path=$model_dir/final.pt

python wenet/bin/recognize_pinyin.py --gpu -1 \
    --mode attention_rescoring \
    --config $model_dir/train.yaml \
    --test_data format.data.tmp \
    --checkpoint $model_path \
    --beam_size 3 \
    --batch_size 1 \
    --penalty 0.0 \
    --dict $dict \
    --result_file text_wav

rm format.data.tmp