# -*- coding: utf-8 -*-

"""
输入类似OpenTransformer准备的text、wav.scp数据格式，其中的转录文本为拼音。
输出wenet的format.data格式。
输出字段中的feat等内容原本是Kaldi对音频提取的特征表示，但这里不进行特征提取，保留原始wav信息。
"""
import argparse
import wave


def parse_kv_file(fp):
    print(f'{parse_kv_file.__name__}: {fp}')
    out = {}
    with open(fp) as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                break
            k, v = line.split(maxsplit=1)
            out[k] = v
    return out


def compute_wav_len(fp):
    with wave.open(fp) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert w.getsampwidth() == 2
        return w.getnframes() / w.getframerate()


def token2tokenid(text, vocabulary):
    tokenids = []
    text_out = []
    unk_id = str(1)
    for x in text.split():
        if x not in vocabulary:
            print(f'OOV {x}')
            tokenids.append(unk_id)
            text_out.append('<unk>')
        else:
            tokenids.append(vocabulary[x])
            text_out.append(x)
    return ' '.join(text_out), tokenids


def main():
    sum_wav_len = 0
    with open(output_data_file, 'w') as outf:
        vocabulary = parse_kv_file(dict_file)
        texts = parse_kv_file(text_file)
        wavs = parse_kv_file(wav_scp_file)
        voc_size = len(vocabulary)
        for k in texts:
            if k in wavs:
                utt = k
                wav_path = wavs[k]
                wav_len = compute_wav_len(wav_path)
                if not min_input_len < wav_len < max_input_len:
                    print(f'Pass wav len {wav_len} of {wav_path}')
                else:
                    text = texts[k]
                    token = text
                    token, tokenid_list = token2tokenid(token, vocabulary)
                    tokenid = ' '.join(tokenid_list)
                    token_shape = f'{len(tokenid_list)},{voc_size}'
                    out_line = f'utt:{utt}\tfeat:{wav_path}\tfeat_shape:{wav_len}\ttext:{text}\ttoken:{token}\ttokenid:{tokenid}\ttoken_shape:{token_shape}'
                    outf.write(out_line + '\n')
                    sum_wav_len += wav_len
    print(f'Generated {output_data_file}')
    print(f'Wav total: {int(sum_wav_len)}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare wenet format data')
    parser.add_argument('-d', help='train|dev')
    args = parser.parse_args()

    feat_dir = 'raw_wav'
    dataset = args.d
    min_input_len = 0.5  # 单位为second
    max_input_len = 16
    # dict_file 示例
    # <blank> 0
    # <unk> 1
    # a1 2
    # <sos/eos> 3
    dict_file = 'data/dict/lang_char.txt'  # vocabulary
    text_file = f'{feat_dir}/{dataset}/text'  # 音频文件id对应拼音转录
    wav_scp_file = f'{feat_dir}/{dataset}/wav.scp'  # 音频文件id对应文件路径
    # format.data 示例
    # utt:BAC009S0916W0372	feat:/home/ubuntu/Data/asr/data_aishell/wav/test/S0916/BAC009S0916W0372.wav	feat_shape:3.3519375	text:湖北一中学教师体罚学生致重伤被判刑三年	token:湖 北 一 中 学 教 师 体 罚 学 生 致 重 伤 被 判 刑 三 年	tokenid:2175 413 2 30 949 1628 1125 162 2956 949 2476 3117 3828 144 3386 350 341 7 1149	token_shape:19,4233
    output_data_file = f'{feat_dir}/{dataset}/format.data'
    main()
