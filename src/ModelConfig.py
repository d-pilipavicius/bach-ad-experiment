import os
import json
from enum import Enum

from torch.cuda import is_available as gpu_avail

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class ModelConfig:
  _instance = None
  _initialized = False

  def __new__(cls, setup_name: str = None):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  
  def __init__(self, setup_name: str | None = None):
    if not self.__class__._initialized:
      setup_name = setup_name or "default"
      with open(f"{BASE_DIR}/config.json") as f:
        config = json.load(f)
      setup = _get_setup(setup_name, config)

      if setup == None:
        raise ValueError(f"No configuration under name \"{setup_name}\" in config.json found")

      # General values
      self.use_cuda = config["cuda"] and gpu_avail()
      self.worker_count = config["worker_count"]
      self.ds_path = config.get("dataset_path")
      self.batch_size = config["batch_size"]
      self.model_dir = config.get("model_dir") or "results" # Anomalib groups models deep inside files, this is only used to change the way it behaves when depositing the model after training
      self.output_dir = config["output_dir"]

      # Setup values
      self.setup_name = setup["name"]
      self.model = ModelType(setup.get("model").upper()) if setup.get("model") else ModelType.PATCHCORE # TODO: Add selected type of model as default
      self.category = setup["category"]
      self.img_count = setup.get("image_count")
      self.random_samples = setup.get("use_random_images")
      self.model_src = setup.get("model_src") # Used for testing/running 
      self.image_w = setup.get("image_w")
      self.image_h = setup.get("image_h")
      self.threshold = setup.get("threshold")
      self.max_epochs = setup.get("max_epochs")

      self.__class__._initialized = True

class ModelType(Enum):
  PATCHCORE = "PATCHCORE"

def _get_setup(setup_name: str, config: dict) -> dict:
  return next((item for item in config["setup"] if item["name"].lower() == setup_name.lower()), None)