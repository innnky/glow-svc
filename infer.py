import time

import librosa
import soundfile
import torch

import models
import utils
from feature_extractor import contentvec768
from vocos import Vocos
from f0_extractor.FCPEF0Predictor import FCPEF0Predictor
import torch.nn.functional as F

config_path = 'configs/config.json'
model_path = '../../Downloads/G_1260.pth'
device = 'cpu'
path = '../../Downloads/source.wav'
spk = 'sanren'
shift = -12


t = time.time()
def calc_time(msg):
    global t
    print(msg, 'time', time.time()-t)
    t = time.time()

hps = utils.get_hparams_from_file(config_path)
model = models.FlowGenerator(
    n_vocab=0,
    out_channels=hps.data.n_mel_channels,
    **hps.model).to(device)
utils.load_checkpoint(model_path, model, None, False)
vocoder = Vocos.from_pretrained('pretrain/vocos/config.yaml', 'pretrain/vocos/pytorch_model.bin').to(device)
content_model = contentvec768.get_model().to(device)
fcpe = FCPEF0Predictor(sampling_rate=16000, hop_length=320, device=device)

calc_time('models load')
t0 = t
wav16k, sr = librosa.load(path, sr=16000)
wav16k = torch.from_numpy(wav16k).to(device)
content = contentvec768.get_content(content_model, wav_16k_tensor=wav16k)
calc_time('content-vec')

pred_f0, uv = fcpe.compute_f0_uv(wav16k)

pred_f0 = fcpe.repeat_expand(pred_f0, int(content.shape[-1]*1.875))
content = fcpe.repeat_expand(content, int(content.shape[-1]*1.875))

calc_time('fcpe f0')

f0 = torch.FloatTensor(pred_f0).unsqueeze(0).to(device)
spk = torch.LongTensor([hps.data.spk2id[spk]]).to(device)
mel_flow, pred_f0 = model(content, f0=f0*(2**(shift/12)), g=spk, gen=True, glow=True)
calc_time('glow-tts')

y_flow = vocoder.decode(mel_flow)
calc_time('vocos vocoder')
print("total time", time.time()-t0)
soundfile.write('../../Downloads/out.wav', y_flow[0].cpu().numpy(), hps.data.sampling_rate)