from metrics.metrics import get_evaluator
from utils.io import get_model_name, failed_loading_model

import torch
from anomalib.models import EfficientAd as _EfficientAd

class EfficientAd(_EfficientAd):
  def test_step(self, batch, batch_idx, *args, **kwargs):
    output = super().test_step(batch, batch_idx, *args, **kwargs)

    if output and output.pred_score:
      output.pred_score = output.pred_score.squeeze(1)

    return output

def load_efficientad(pretrained: bool = False) -> EfficientAd:
  evaluator = get_evaluator()
  model = EfficientAd(
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