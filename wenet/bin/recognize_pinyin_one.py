# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import time

import torch
import torchaudio
import yaml
from torchaudio.compliance import kaldi

from wenet.dataset.dataset import CollateFunc
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint


def wav_feat(wav_file, feature_extraction_conf):
    waveform, sample_rate = torchaudio.load_wav(wav_file)
    wav_dither = 1.0

    mat = kaldi.fbank(
        waveform,
        num_mel_bins=feature_extraction_conf['mel_bins'],
        frame_length=feature_extraction_conf['frame_length'],
        frame_shift=feature_extraction_conf['frame_shift'],
        dither=wav_dither,
        energy_floor=0.0,
        sample_frequency=sample_rate
    )
    feat = mat
    length = mat.shape[0]
    return feat.unsqueeze(0), torch.tensor([length])


if __name__ == '__main__':
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
    parser.add_argument('--result_file', required=True, help='asr result file')
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
    args = parser.parse_args()
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

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin)

    raw_wav = configs['raw_wav']
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

    feature_extraction_conf = test_collate_conf['feature_extraction_conf']

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

    start = time.time()
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        feats, feats_lengths = wav_feat(args.test_data, feature_extraction_conf)
        # print(f'feats.shape {feats.shape}, feats_lengths.shape {feats_lengths.shape}')
        feats = feats.to(device)
        feats_lengths = feats_lengths.to(device)
        if args.mode == 'attention':
            hyps = model.recognize(
                feats,
                feats_lengths,
                beam_size=args.beam_size,
                decoding_chunk_size=args.decoding_chunk_size,
                num_decoding_left_chunks=args.num_decoding_left_chunks,
                simulate_streaming=args.simulate_streaming)
            hyps = [hyp.tolist() for hyp in hyps]
        elif args.mode == 'ctc_greedy_search':
            hyps = model.ctc_greedy_search(
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
            hyp = model.ctc_prefix_beam_search(
                feats,
                feats_lengths,
                args.beam_size,
                decoding_chunk_size=args.decoding_chunk_size,
                num_decoding_left_chunks=args.num_decoding_left_chunks,
                simulate_streaming=args.simulate_streaming)
            hyps = [hyp]
        elif args.mode == 'attention_rescoring':
            assert (feats.size(0) == 1)
            hyp = model.attention_rescoring(
                feats,
                feats_lengths,
                args.beam_size,
                decoding_chunk_size=args.decoding_chunk_size,
                num_decoding_left_chunks=args.num_decoding_left_chunks,
                ctc_weight=args.ctc_weight,
                simulate_streaming=args.simulate_streaming)
            hyps = [hyp]
        for i, key in enumerate(range(len(feats))):
            content = []
            for w in hyps[i]:
                if w == eos:
                    break
                content.append(char_dict[w])
            # 输出为拼音，所以用空格分隔才好识别与计算CER
            content = ' '.join(content)
            logging.info('{} {}'.format(key, content))
            fout.write('{} {}\n'.format(key, content))
    logging.info(f'Elapse: {time.time() - start:.3}s\n')
