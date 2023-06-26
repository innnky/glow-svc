import tqdm
from text.cleaner import g2p, text_normalize

transcription_path = 'filelists/transcription.list'
cleaned_path = transcription_path + '.cleaned'
with open(cleaned_path, 'w', encoding='utf-8') as f:
    for line in tqdm.tqdm(open(transcription_path, encoding='utf-8').readlines()):
        utt, spk, text = line.strip().split('|')
        try:
            text = text_normalize(text)
            phones, tones, word2ph = g2p(norm_text=text)
            phones = ['_']+ phones +['_']
            tones = [0]+ tones +[0]
            f.write('{}|{}|{}|{}\n'.format(utt, spk, ' '.join(phones), " ".join([str(i) for i in tones])))

        except:
            print("skip:",utt, spk, text)

# 划分训练集测试集，保存至filelists/train.list和filelists/val.list
# shuffle
data_all = [line for line in open(cleaned_path).readlines()]
import random
random.shuffle(data_all)
data_train = data_all[:-5]
data_val = data_all[-5:]
with open('filelists/train.list', 'w', encoding='utf-8') as f:
    for line in data_train:
        f.write(line)

with open('filelists/val.list', 'w', encoding='utf-8') as f:
    for line in data_val:
        f.write(line)
