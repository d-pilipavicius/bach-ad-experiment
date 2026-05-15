from ModelConfig import ModelType
from model.patchcore import load_patchcore, load_dual_patchcore
from model.efficientad import load_efficientad
from model.cfa import load_cfa

from anomalib.models import AnomalibModule

def get_model(type: ModelType, pretrained: bool = False) -> AnomalibModule:
  match(type):
    case ModelType.PATCHCORE:
      model = load_patchcore(pretrained)
    case ModelType.EFFICIENTAD:
      model = load_efficientad(pretrained)
    case ModelType.CFA:
      model = load_cfa(pretrained)
    case ModelType.DUALPATCHCORE: 
      model = load_dual_patchcore(pretrained)

  return model