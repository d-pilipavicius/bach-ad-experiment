import sys
from enum import Enum
from pathlib import Path

from ModelConfig import ModelConfig

class RunType(Enum):
  TRAIN = "TRAIN"
  TEST = "TEST"
  IMAGE = "IMAGE"

def read_args() -> tuple[RunType, Path | None]:
  argv = sys.argv

  # If no params
  if len(sys.argv) == 1:
    _print_info()
  
  # Load in TRAIN/TEST/IMAGE
  runtype = _arg_1(argv) 
  
  # Additional params
  flags = _flag_parser(argv[2:])
  _validate_flags(runtype, flags)
      
  # Init ModelConfig singleton
  ModelConfig(flags["name"]["data"])

  return (runtype, flags["image"]["data"])

def _arg_1(argv: list[str]) -> RunType:
  upper = argv[1].upper()
  if upper == '-H' or upper == '--HELP':
    _print_info(argv[0])
  else:
    return RunType(upper)
  
def _init_flags() -> dict:
  return {
    "name": {
      "loaded": False,
      "data": None
    },
    "image": {
      "loaded": False,
      "data": None
    }
  }

def _flag_parser(argv: list[str]) -> dict:
  flags = _init_flags()

  while len(argv) > 0:
    match argv[0]:
      case "-n":
        _load_setup_name(flags, argv)
        argv = argv[2:]
      case "-i":
        load_test_image(flags, argv)
        argv = argv[2:]
      case _:
        _unrecognised_command(argv[0])

  return flags

def _load_setup_name(flags: dict, argv: list[str]) -> None:
  if len(argv) < 2:
    _missing_setup_name()
  if flags["name"]["loaded"]:
    _duplicate_flag("-n")

  flags["name"]["loaded"] = True
  flags["name"]["data"] = argv[1]

def load_test_image(flags: dict, argv: list[str]) -> None:
  if len(argv) < 2:
    _missing_image_path()
  if flags["image"]["loaded"]:
    _duplicate_flag("-i")

  flags["image"]["loaded"] = True
  flags["image"]["data"] = Path(argv[1])

def _validate_flags(runtype: RunType, flags: dict):
  if runtype is RunType.IMAGE and not flags["image"]["loaded"]:
    _run_model_with_no_image()

def _print_info() -> None:
  print(
    f"""
    Author: Domantas Pilipavicius
    This runs experiments related to my bachelors thesis.
    Usage: python main.py (TRAIN|TEST|IMAGE)
  
    Startup:
      -n CONFIG_NAME            Starts code using specific "setup" from config.json file. 
                                CONFIG_NAME is case insensitive. By default, CONFIG_NAME=default.
      -i IMAGE_PATH             When used with runtype IMAGE, imports image from filepath, runs
                                it through the model, produced the output in "output_dir" directory
                                defined in config.json.
    """
  )
  sys.exit(0)

def _unrecognised_command(command: str) -> None:
  raise ValueError(f"Unrecognised command {command}, run: python main.py --help")

def _missing_setup_name() -> None:
  raise ValueError(f"Missing configuration name on flag, usage: -n SETUP_NAME")

def _missing_image_path() -> None:
  raise ValueError(f"Missing image path on flag, usage: -i IMAGE_PATH")

def _duplicate_flag(flag: str) -> None:
  raise ValueError(f"Duplicate {flag} flag detected")

def _run_model_with_no_image() -> None:
  raise ValueError(f"To use main.py IMAGE, the -i flag must be defined with an image.")