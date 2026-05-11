from utils.io import get_model_name, failed_loading_model
from metrics.metrics import get_evaluator

import torch
from anomalib.models import Patchcore

# If source left as empty, model is not pre-trained
def load_patchcore(pretrained: bool = False) -> Patchcore:
  evaluator = get_evaluator()
  model = Patchcore(
    coreset_sampling_ratio=0.1,
    evaluator=evaluator,
  )

  if pretrained:
    ckpt_path = get_model_name()
    try:
      ckpt = torch.load(ckpt_path, weights_only=False)
    except Exception as e:
      failed_loading_model(e)
    model.load_state_dict(ckpt["state_dict"], strict=False)
  
  return model