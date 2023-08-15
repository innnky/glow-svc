import os

from torch.utils.data import DataLoader

import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from tqdm import tqdm
import logging
logging.getLogger('numba').setLevel(logging.INFO)
config_path = 'configs/config.json'
hps = utils.get_hparams_from_file(config_path)
collate = TextAudioSpeakerCollate()
train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)

for _ in tqdm(train_dataset):
    pass
for _ in tqdm(eval_dataset):
    pass

# train_loader = DataLoader(train_dataset, num_workers=0, shuffle=False,
#                             batch_size=2, pin_memory=True,
#                             drop_last=True, collate_fn=collate)
#
# for _ in tqdm(train_loader):
#     pass