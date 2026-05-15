from utils.io import get_model_name, failed_loading_model

import torch
from anomalib.models import AnomalibModule

def init_pretrained_model(model: AnomalibModule) -> None:
  ckpt_path = get_model_name()
  try:
    ckpt = torch.load(ckpt_path, weights_only=False)
  except Exception as e:
    failed_loading_model(e)
  model.load_state_dict(ckpt["state_dict"], strict=False)