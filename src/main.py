from args import RunType, read_args
from model.get_model import get_model
from ModelConfig import ModelConfig, ModelType
from model.run_model import train_model, test_model, run_model, train_dual_patchcore

def main() -> None:
  run_type, image_path = read_args()
  config = ModelConfig()
  model = get_model(config.model, run_type is not RunType.TRAIN)

  match run_type:
    case RunType.TRAIN:
      if config.model is ModelType.DUALPATCHCORE:
        train_dual_patchcore(model)
      else:
        train_model(model)
    case RunType.TEST:
      test_model(model)
    case RunType.IMAGE:
      run_model(model, image_path)

if __name__ == "__main__":
  main()