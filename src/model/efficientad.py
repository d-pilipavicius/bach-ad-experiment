from metrics.metrics import get_evaluator
from utils.model import init_pretrained_model

from anomalib.models import EfficientAd as _EfficientAd

class EfficientAd(_EfficientAd):
  def test_step(self, batch, batch_idx, *args, **kwargs):
    output = super().test_step(batch, batch_idx, *args, **kwargs)

    if output and output.pred_score:
      output.pred_score = output.pred_score.squeeze(1)

    return output
  
  def predict_step(self, batch, batch_idx, *args, **kwargs):
    output = super().predict_step(batch, batch_idx, *args, **kwargs)

    if output and output.pred_score:
      output.pred_score = output.pred_score.squeeze(1)

    return output

def load_efficientad(pretrained: bool = False) -> EfficientAd:
  evaluator = get_evaluator()
  model = EfficientAd(
    evaluator=evaluator,
  )

  if pretrained:
    init_pretrained_model(model)

  return model