

import os

import torch
from tqdm import tqdm

x = {}
x["min"] = torch.FloatTensor([[[-5]]])
x["max"] = torch.FloatTensor([[[6]]])
torch.save(x, 'configs/duration_stats.pt')

mel_dim = 128
mel_min = torch.ones(1, mel_dim) * float('inf')
mel_max = torch.ones(1, mel_dim) * -float('inf')

for spk in tqdm(os.listdir(f"dataset/")):
    if os.path.isdir(f"dataset/{spk}"):
        for pt in tqdm(os.listdir(f"dataset/{spk}")):
            # try:
            if not pt.endswith("mel.pt"):
                continue
            mel = torch.load(f"dataset/{spk}/{pt}", "cpu").unsqueeze(0)

            s_min = mel.min(dim=2)[0]
            s_max = mel.max(dim=2)[0]

            mel_min = torch.minimum(s_min, mel_min)
            mel_max = torch.maximum(s_max, mel_max)
            #
            # except:
            #     print(spk, pt, "failed")

mel_min = mel_min.unsqueeze(-1)
mel_max = mel_max.unsqueeze(-1)
print(mel_min)
print(mel_max)

torch.save({
    "min": mel_min,
    "max": mel_max
}, "configs/mel_stats.pt")




pitch_min = torch.ones(1) * float('inf')
pitch_max = torch.ones(1) * -float('inf')

for spk in tqdm(os.listdir(f"dataset/")):
    if os.path.isdir(f"dataset/{spk}"):
        for pt in tqdm(os.listdir(f"dataset/{spk}")):
            if not pt.endswith("f0.pt"):
                continue
            f0 = torch.load(f"dataset/{spk}/{pt}", "cpu")
            f0 = 2595. * torch.log10(1. + f0 / 700.) / 500

            min = f0.min()
            max = f0.max()
            pitch_min = torch.minimum(min, pitch_min)
            pitch_max = torch.maximum(max, pitch_max)

print(pitch_min)
print(pitch_max)
torch.save({
    "min": pitch_min.unsqueeze(0).unsqueeze(0),
    "max": pitch_max.unsqueeze(0).unsqueeze(0)
}, "configs/pitch_stats.pt")
