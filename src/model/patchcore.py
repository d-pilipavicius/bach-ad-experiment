from metrics.metrics import get_evaluator
from utils.model import init_pretrained_model

from anomalib.models import Patchcore

from .dual_patchcore.lightning_model import DualPatchcore

def load_patchcore(pretrained: bool = False) -> Patchcore:
  evaluator = get_evaluator()
  model = Patchcore(
    coreset_sampling_ratio=0.1,
    evaluator=evaluator,
  )

  if pretrained:
    init_pretrained_model(model)
  
  return model

def load_dual_patchcore(pretrained: bool = False) -> DualPatchcore:
  evaluator = get_evaluator()
  model = DualPatchcore(
    coreset_sampling_ratio=0.1,
    evaluator=evaluator,
  )

  if pretrained:
    init_pretrained_model(model)
  
  return model