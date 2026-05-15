from metrics.metrics import get_evaluator
from utils.model import init_pretrained_model

from anomalib.models import Cfa

def load_cfa(pretrained: bool = False) -> Cfa:
  evaluator = get_evaluator()
  model = Cfa(
    evaluator=evaluator,
  )

  if pretrained:
    init_pretrained_model(model)
  
  return model