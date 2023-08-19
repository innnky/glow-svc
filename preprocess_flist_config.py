import json

import tqdm

import os
data_all = []
spk2id = {}
current_spk = 0
for spk in os.listdir('dataset'):
    if os.path.isdir(os.path.join('dataset', spk)):
        for wav in os.listdir(os.path.join('dataset', spk)):
            if wav.endswith('wav'):
                name = wav.split('.')[0]
                data_all.append(f"{name}|{spk}\n")
                if spk not in spk2id.keys():
                    spk2id[spk] = current_spk
                    current_spk+=1


import random
# random.shuffle(data_all)
data_all = sorted(data_all)
data_train = data_all[:-5]
data_val = data_all[-5:]
with open('filelists/train.list', 'w', encoding='utf-8') as f:
    for line in data_train:
        f.write(line)

with open('filelists/val.list', 'w', encoding='utf-8') as f:
    for line in data_val:
        f.write(line)

template = json.load(open('configs/config.json', 'r', encoding='utf-8'))
template["data"]['spk2id'] = spk2id
json.dump(template, open('configs/config.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

