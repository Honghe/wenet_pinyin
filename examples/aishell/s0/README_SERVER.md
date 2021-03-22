# Wenet as ASR server
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
