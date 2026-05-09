import uuid
import shutil
from enum import Enum
from pathlib import Path
from datetime import datetime

from ModelConfig import ModelConfig

class FileType(Enum):
  TEST_DATA = "TEST_DATA"
  IMAGE = "IMAGE"

def move_model(filepath: Path | str) -> None:
  if not isinstance(filepath, Path):
    filepath = Path(filepath)
  dest = _get_output_dir()
  old_model = dest / filepath.name

  if old_model.exists():
    renamed = old_model.with_name(f"old_model_{uuid.uuid4()}.ckpt")
    old_model.rename(renamed)
  
  shutil.move(str(filepath), str(dest))

def write_file(type: FileType, data: any) -> None:
  path = _get_output_dir() / _get_filename(type)
  match(type):
    case FileType.TEST_DATA:
      # data is string
      with(path) as f:
        f.write_text(data) 
    case FileType.IMAGE:
      # data is PIL image
      data.save(path)

def get_model_name() -> str:
  config = ModelConfig()
  if config.model_src is None:
    path = _get_output_dir() / "model.ckpt"
  else:
    path = config.model_src

  return path

def _get_filename(type: FileType) -> str:
  now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
  match(type):
    case FileType.TEST_DATA:
      filename = f"model_test_{now}.txt"
    case FileType.IMAGE:
      filename = f"model_image_{now}.png"

  return filename

def _get_output_dir() -> Path:
  config = ModelConfig()
  setup_path = (Path(config.output_dir) / 
    config.model.name.lower() / 
    config.category.lower() /
    config.setup_name.lower())
  setup_path.mkdir(exist_ok=True, parents=True)
  return setup_path