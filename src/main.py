from args import RunType, read_args
from model.patchcore import load_patchcore
from model.run_model import train_model, test_model, run_model

def main() -> None:
  run_type, image_path = read_args()

  match run_type:
    case RunType.TRAIN:
      train_model(load_patchcore())
    case RunType.TEST:
      test_model(load_patchcore(pretrained=True))
    case RunType.IMAGE:
      run_model(load_patchcore(pretrained=True), image_path)

if __name__ == "__main__":
  main()