# Performance Record

## Conformer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.002, batch size 18, 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 919f07c4887ac500168ba84b39b535fd8e58918a
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210204_conformer_exp.tar.gz

| decoding mode          | CER  |
|------------------------|------|
| attention decoder      | 5.18 |
| ctc greedy search      | 4.94 |
| ctc prefix beam search | 4.94 |
| attention rescoring    | 4.61 |

## Unified Conformer Result

* Feature info: using fbank feature, dither=0, cmvn, oneline speed perturb
* Training info: lr 0.001, batch size 16, 8 gpu, acc_grad 1, 180 epochs, dither 0.0
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 919f07c4887ac500168ba84b39b535fd8e58918a
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210203_unified_conformer_exp.tar.gz

| decoding mode/chunk size | full | 16   | 8    | 4    |
|--------------------------|------|------|------|------|
| attention decoder        | 5.40 | 5.60 | 5.74 | 5.86 |
| ctc greedy search        | 5.56 | 6.29 | 6.68 | 7.10 |
| ctc prefix beam search   | 5.57 | 6.30 | 6.67 | 7.10 |
| attention rescoring      | 5.05 | 5.45 | 5.69 | 5.91 |

## Transformer Result

* Feature info: using fbank feature, dither, with cmvn, no speed perturb.
* Training info: lr 0.002, batch size 26, 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 919f07c4887ac500168ba84b39b535fd8e58918a
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210204_transformer_exp.tar.gz

| decoding mode          | CER  |
|------------------------|------|
| attention decoder      | 5.69 |
| ctc greedy search      | 5.92 |
| ctc prefix beam search | 5.91 |
| attention rescoring    | 5.30 |

## Unified Transformer Result

* Feature info: using fbank feature, dither=0, with cmvn, no speed perturb.
* Training info: lr 0.002, batch size 16, 8 gpu, acc_grad 1, 120 epochs, dither 0.0
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 919f07c4887ac500168ba84b39b535fd8e58918a
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210204_unified_transformer_exp.tar.gz

| decoding mode/chunk size | full | 16   | 8    | 4    |
|--------------------------|------|------|------|------|
| attention decoder        | 6.13 | 6.43 | 6.55 | 6.79 |
| ctc greedy search        | 6.73 | 7.99 | 8.72 | 9.92 |
| ctc prefix beam search   | 6.73 | 7.99 | 8.73 | 9.91 |
| attention rescoring      | 5.80 | 6.56 | 7.02 | 7.68 |

## Recognize one wav file demo
```
. ./reg_wav.sh <wav_file>
```
为使其识别结果为拼音，几个相关的修改添加的文件如下，所以上游有更新时，这几个文件需要port。
```
s0/conf/train_unified_conformer_pinyin.yaml
run_pinyin.sh
wenet/bin/recognize_pinyin.py
```

## Huey and Redis
### Start
Use `OMP_NUM_THREADS` to set Pytorch process affinity.
```
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 huey_consumer.py  sev_huey.huey -w 4
```

### disable all data persistence
To disable all data persistence in Redis do the following in the redis.conf file:

Disable AOF by setting the appendonly configuration directive to no (it is the default value). like this:
```
appendonly no
```
Disable RDB snapshotting by commenting all of the save configuration directives (there are 3 that are defined by default)
```
#save 900 1
#save 300 10
#save 60 10000
```
After change, make sure you restart Redis to apply them.
```
systemctl restart redis
```
### Prevent last Huey process not killed
Kill the last process before running new huey_comsumer.py
