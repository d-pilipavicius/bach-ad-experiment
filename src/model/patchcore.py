from metrics.metrics import get_evaluator

import torch
from anomalib.models import Patchcore

# If source left as empty, model is not pre-trained
def load_patchcore(src_filepath: str | None = None) -> Patchcore:
  evaluator = get_evaluator()
  model = Patchcore(
    coreset_sampling_ratio=0.1,
    evaluator=evaluator,
  )

  if src_filepath is not None:
    ckpt = torch.load(src_filepath, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
  
  return model