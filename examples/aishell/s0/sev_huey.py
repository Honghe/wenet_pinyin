# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import time

import torch
import yaml
from huey import RedisHuey
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader

from wenet.dataset.dataset import AudioDatasetPinyin, CollateFunc
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint

format_ = "%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(message)s"
logging.basicConfig(format=format_, level=logging.DEBUG)

class ASR:
    def init(self):
        with open(args.config, 'r') as fin:
            configs = yaml.load(fin)

        raw_wav = configs['raw_wav']
        # Init dataset and data loader
        # Init dataset and data loader
        test_collate_conf = copy.deepcopy(configs['collate_conf'])
        test_collate_conf['spec_aug'] = False
        test_collate_conf['spec_sub'] = False
        test_collate_conf['feature_dither'] = False
        test_collate_conf['speed_perturb'] = False
        if raw_wav:
            test_collate_conf['wav_distortion_conf']['wav_distortion_rate'] = 0
        test_collate_func = CollateFunc(**test_collate_conf,
                                        raw_wav=raw_wav)
        dataset_conf = configs.get('dataset_conf', {})
        dataset_conf['batch_size'] = args.batch_size
        dataset_conf['batch_type'] = 'static'
        dataset_conf['sort'] = False

        # Init asr model from configs
        model = init_asr_model(configs)

        # Load dict
        char_dict = {}
        with open(args.dict, 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                char_dict[int(arr[1])] = arr[0]
        eos = len(char_dict) - 1

        load_checkpoint(model, args.checkpoint)
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)

        model.eval()

        self.dataset_conf = dataset_conf
        self.test_collate_func = test_collate_func
        self.device = device
        self.model = model
        self.char_dict = char_dict
        self.eos = eos
        self.raw_wav = raw_wav

    def reg(self, data):
        start = time.time()
        d = "utt:1\tfeat:/home/ubuntu/Data/asr/audio_command/6seconds.wav\tfeat_shape:5\ttext:一\ttoken:一\ttokenid:1\ttoken_shape:1,1"
        test_dataset = AudioDatasetPinyin(d, **self.dataset_conf, raw_wav=self.raw_wav)
        test_data_loader = DataLoader(test_dataset,
                                      collate_fn=self.test_collate_func,
                                      shuffle=False,
                                      batch_size=1,
                                      num_workers=0)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data_loader):
                keys, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(self.device)
                feats_lengths = feats_lengths.to(self.device)
                if args.mode == 'attention':
                    hyps = self.model.recognize(
                        feats,
                        feats_lengths,
                        beam_size=args.beam_size,
                        decoding_chunk_size=args.decoding_chunk_size,
                        num_decoding_left_chunks=args.num_decoding_left_chunks,
                        simulate_streaming=args.simulate_streaming)
                    hyps = [hyp.tolist() for hyp in hyps]
                elif args.mode == 'ctc_greedy_search':
                    hyps = self.model.ctc_greedy_search(
                        feats,
                        feats_lengths,
                        decoding_chunk_size=args.decoding_chunk_size,
                        num_decoding_left_chunks=args.num_decoding_left_chunks,
                        simulate_streaming=args.simulate_streaming)
                # ctc_prefix_beam_search and attention_rescoring only return one
                # result in List[int], change it to List[List[int]] for compatible
                # with other batch decoding mode
                elif args.mode == 'ctc_prefix_beam_search':
                    assert (feats.size(0) == 1)
                    hyp = self.model.ctc_prefix_beam_search(
                        feats,
                        feats_lengths,
                        args.beam_size,
                        decoding_chunk_size=args.decoding_chunk_size,
                        num_decoding_left_chunks=args.num_decoding_left_chunks,
                        simulate_streaming=args.simulate_streaming)
                    hyps = [hyp]
                elif args.mode == 'attention_rescoring':
                    assert (feats.size(0) == 1)
                    hyp = self.model.attention_rescoring(
                        feats,
                        feats_lengths,
                        args.beam_size,
                        decoding_chunk_size=args.decoding_chunk_size,
                        num_decoding_left_chunks=args.num_decoding_left_chunks,
                        ctc_weight=args.ctc_weight,
                        simulate_streaming=args.simulate_streaming)
                    hyps = [hyp]
                for i, key in enumerate(keys):
                    content = []
                    for w in hyps[i]:
                        if w == self.eos:
                            break
                        content.append(self.char_dict[w])
                    # 输出为拼音，所以用空格分隔才好识别与计算CER
                    content = ' '.join(content)
                    logging.info('{} {}'.format(key, content))
        logging.info(f'Elapse: {time.time() - start:.3}s')
        return content


parser = argparse.ArgumentParser(description='recognize with your model')
parser.add_argument('--config', required=True, help='config file')
parser.add_argument('--test_data', required=True, help='test data file')
parser.add_argument('--gpu',
                    type=int,
                    default=-1,
                    help='gpu id for this rank, -1 for cpu')
parser.add_argument('--checkpoint', required=True, help='checkpoint model')
parser.add_argument('--dict', required=True, help='dict file')
parser.add_argument('--beam_size',
                    type=int,
                    default=10,
                    help='beam size for search')
parser.add_argument('--penalty',
                    type=float,
                    default=0.0,
                    help='length penalty')
parser.add_argument('--batch_size',
                    type=int,
                    default=16,
                    help='asr result file')
parser.add_argument('--mode',
                    choices=[
                        'attention', 'ctc_greedy_search',
                        'ctc_prefix_beam_search', 'attention_rescoring'],
                    default='attention',
                    help='decoding mode')
parser.add_argument('--ctc_weight',
                    type=float,
                    default=0.0,
                    help='ctc weight for attention rescoring decode mode')
parser.add_argument('--decoding_chunk_size',
                    type=int,
                    default=-1,
                    help='''decoding chunk size,
                            <0: for decoding, use full chunk.
                            >0: for decoding, use fixed chunk size as set.
                            0: used for training, it's prohibited here''')
parser.add_argument('--num_decoding_left_chunks',
                    type=int,
                    default=-1,
                    help='number of left chunks for decoding')

parser.add_argument('--simulate_streaming',
                    action='store_true',
                    help='simulate streaming inference')

#
dict = "data/dict/lang_char.txt"
model_dir = "exp/train_unified_conformer/"
model_path = f"{model_dir}/final.pt"

args = parser.parse_args(f"""--gpu -1
    --mode attention_rescoring
    --config {model_dir}/train.yaml
    --test_data format.data.tmp
    --checkpoint {model_path}
    --beam_size 3
    --batch_size 1
    --penalty 0.0
    --dict {dict}""".split())

print(args)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring'
                 ] and args.batch_size > 1:
    logging.fatal(
        'decoding mode {} must be running with batch_size == 1'.format(
            args.mode))
    sys.exit(1)

try:
    set_start_method('spawn')
except RuntimeError as e:
    print(e)

# Huey
huey = RedisHuey()

asr = ASR()


@huey.on_startup()
def new_asr():
    asr.init()


@huey.task()
def hello(msg):
    return msg


@huey.task()
def reg(data, timestamp):
    if time.time() - timestamp > 0.5:
        logging.info('Task timestamp expired.')
        return 'None'
    else:
        return asr.reg(data)
