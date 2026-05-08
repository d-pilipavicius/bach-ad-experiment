import os
import json

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
      setup = get_setup(setup_name, config)

      if setup == None:
        raise ValueError(f"No configuration under name \"{setup_name}\" in config.json found")

      # General values
      self.use_cuda = config["cuda"] and gpu_avail()
      self.worker_count = config["worker_count"]
      self.threshold = config["threshold"]
      self.ds_path = config.get("dataset_path")
      self.batch_size = config["batch_size"]
      
      # Training setup values
      self.setup_name = setup["name"]
      self.category = setup["category"]
      self.img_count = setup.get("image_count")
      self.random_samples = setup.get("use_random_images")
      self.model_filename = setup["model_filename"]
      self.image_w = setup.get("image_w")
      self.image_h = setup.get("image_h")

      self.__class__._initialized = True

def get_setup(setup_name, config):
  return next((item for item in config["setup"] if item["name"].lower() == setup_name.lower()), None)