import os
import json

from torch.cuda import is_available as gpu_avail

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class ModelConfig:
  _instance = None
  _initialized = False

  def __new__(cls, train_setup_name: str = None):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  
  def __init__(self, train_setup_name: str = "default"):
    if not self.__class__._initialized:
      with open(f"{BASE_DIR}/config.json") as f:
        config = json.load(f)
      trn_setup = get_trn_setup(train_setup_name, config)

      if trn_setup == None:
        raise ValueError(f"No configuration under name \"{train_setup_name}\" in config.json found")

      # General values
      self.use_cuda = config["cuda"] and gpu_avail()
      self.worker_count = config["worker_count"]
      self.threshold = config["threshold"]
      self.ds_path = config.get("dataset_path")
      self.batch_size = config["batch_size"]
      
      # Training setup values
      self.trn_setup_name = trn_setup["name"]
      self.category = trn_setup["category"]
      self.img_count = trn_setup.get("image_count")
      self.random_samples = trn_setup.get("use_random_images")
      self.model_filename = trn_setup["model_filename"]
      self.image_w = trn_setup.get("image_w")
      self.image_h = trn_setup.get("image_h")

      self.__class__._initialized = True

def get_trn_setup(train_setup_name, config):
  return next((item for item in config["train_setup"] if item["name"].lower() == train_setup_name.lower()), None)