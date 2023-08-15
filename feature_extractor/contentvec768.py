import torch
from transformers import HubertModel

from torch import nn
import logging

logging.getLogger("numba").setLevel(logging.WARNING)

class HubertModelWithFinalProj(HubertModel):
  def __init__(self, config):
    super().__init__(config)

    # Remove this layer is necessary to achieve the desired outcome.
    self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

def get_model():
  model = HubertModelWithFinalProj.from_pretrained("./pretrain/content-vec-best")
  return model


def get_content(hmodel, wav_16k_tensor):
  with torch.no_grad():
      feats = hmodel(wav_16k_tensor.unsqueeze(0))["last_hidden_state"]
  return feats.transpose(1,2)

