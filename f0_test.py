import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch.nn

import mel_processing
from f0_extractor.FCPEF0Predictor import FCPEF0Predictor

fcpe = FCPEF0Predictor()

path  =  "../../Downloads/individualAudio (61).wav"

wav, sr = librosa.load(path, sr=44100)

pred_f0, uv = fcpe.compute_f0_uv(wav)
pred_f0 = torch.FloatTensor(pred_f0).unsqueeze(0).unsqueeze(-1)
cent = fcpe.fcpe.model.f0_to_cent( pred_f0)
x = fcpe.fcpe.model.gaussian_blurred_cent(cent)

activation = x.transpose(1,2)


# mel2, _ = mel_processing.get_mel(path,44100,128,2048,2048,512,0,16000)
# activation, mel = fcpe.get_activation(wav, down_sample=3)
# activation = torch.nn.functional.interpolate(activation, size=mel2.shape[-1], mode="linear")


print(activation.shape)
activation = activation[0].numpy()
rotated_activation = np.rot90(activation.T, k=1)

plt.imshow(rotated_activation, cmap='hot',aspect="auto",)
plt.tight_layout()
print(rotated_activation.min(), rotated_activation.max())
# plt.axis('off')  # 关闭坐标轴显示

plt.show()
