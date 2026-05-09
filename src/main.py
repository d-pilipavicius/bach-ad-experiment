import sys
from enum import Enum
from pathlib import Path

import torch 
from anomalib import PrecisionType

from utils.io import get_model_name
from ModelConfig import ModelConfig
from model.patchcore import load_patchcore
from model.run_model import train_model, test_model

class RunType(Enum):
  TRAIN = "TRAIN"
  TEST = "TEST"
  IMAGE = "IMAGE"

def main() -> None:
  torch.serialization.add_safe_globals([PrecisionType])
  run_type = read_args()
  
  match run_type:
    case RunType.TRAIN:
      train_model(load_patchcore())
    case RunType.TEST:
      test_model(load_patchcore(get_model_name()))
    case RunType.IMAGE:
      pass

def run(image_path: Path):
  pass

def read_args() -> tuple[RunType, Path | None]:
  argc = len(sys.argv)
  argv = sys.argv

  if argc == 1:
    print_info(argv[0])
  
  split = arg_1(argv) 
  ModelConfig(get_setup_name(argc, argv))
      
  return split

def arg_1(argv: list[str]) -> RunType:
  upper = argv[1].upper()
  if upper == '-H' or upper == '--HELP':
    print_info(argv[0])
  else:
    return RunType(upper)

def get_setup_name(argc: int, argv: str) -> str | None:
  if argc == 2:
    setup_name = None

  elif argv[2].lower() != '-n':
    unrecognised_command(argv[0], argv[2])
  elif argc == 3:
    raise ValueError(f"Missing -n parameter CONFIG_NAME")
  elif argc == 4:
    setup_name = argv[3]
  else:
    unrecognised_command(argv[0], argv[4])

  return setup_name

def print_info(main_filename: str) -> None:
  print(
    f"""
    Author: Domantas Pilipavicius
    This runs experiments related to my bachelors thesis.
    Usage: python {main_filename} (TRAIN|TEST|IMAGE)
  
    Startup:
      -n CONFIG_NAME            Starts code using specific "setup" from config.json file. 
                                CONFIG_NAME is case insensitive. By default, CONFIG_NAME=default.
    """
  )
  sys.exit(0)

def unrecognised_command(main_filename: str, command: str) -> None:
  raise ValueError(f"Unrecognised command {command}, run: python {main_filename} --help")

if __name__ == "__main__":
  main()