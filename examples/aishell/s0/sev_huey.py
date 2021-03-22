# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import time

from huey import RedisHuey

from wenet.bin.recognize_pinyin_one import WeNetASR

format_ = "%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(message)s"
logging.basicConfig(format=format_, level=logging.DEBUG)

# try:
#     set_start_method('spawn')
# except RuntimeError as e:
#     print(e)

# Huey
huey = RedisHuey()

asr = WeNetASR()


@huey.on_startup()
def new_asr():
    dict = "data/dict/lang_char.txt"
    model_dir = "exp/train_unified_conformer/"
    model_path = f"{model_dir}/final.pt"

    args_srt = f"""--gpu -1
        --mode attention_rescoring
        --config {model_dir}/train.yaml
        --test_data format.data.tmp
        --checkpoint {model_path}
        --beam_size 3
        --batch_size 1
        --penalty 0.0
        --dict {dict}
        --result_file dummy_file""".split()

    asr.init_args(args_srt)
    asr.init_model()


@huey.task()
def hello(msg):
    return msg


@huey.task()
def reg(data: bytes, timestamp):
    if time.time() - timestamp > 0.5:
        logging.info('Task timestamp expired, pass.')
        return 'None'
    else:
        return asr.reg(data)
