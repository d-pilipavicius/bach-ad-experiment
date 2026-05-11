from ModelConfig import ModelType
from model.patchcore import load_patchcore
from model.efficientad import load_efficientad

from anomalib.models import AnomalibModule

def get_model(type: ModelType, pretrained: bool = False) -> AnomalibModule:
  match(type):
    case ModelType.PATCHCORE:
      model = load_patchcore(pretrained)
    case ModelType.EFFICIENTAD:
      model = load_efficientad(pretrained)
  
  return model