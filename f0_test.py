import librosa
import matplotlib.pyplot as plt

from f0_extractor.FCPEF0Predictor import FCPEF0Predictor

fcpe = FCPEF0Predictor()

path  =  "../../Downloads/individualAudio (61).wav"

wav, sr = librosa.load(path, sr=44100)

pred_f0, uv = fcpe.compute_f0_uv(wav)
plt.plot(pred_f0*uv)
# plt.plot(uv)
plt.show()
# activation = fcpe.get_activation(wav)