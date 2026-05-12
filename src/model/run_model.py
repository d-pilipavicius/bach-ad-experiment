from utils.dict import opt
from ModelConfig import ModelConfig
from metrics.metrics import format_metrics
from utils.image import output_to_marked_pil
from utils.io import write_file, move_model, FileType
from dataset.SubMVTecLOCO import get_configured_ds, SampleType

from anomalib.engine import Engine
from anomalib.models import AnomalibModule

def train_model(model: AnomalibModule) -> any:
  dataset = get_configured_ds()
  engine = _get_engine()
  output = engine.fit(model, dataset)
  move_model(engine.trainer.checkpoint_callback.best_model_path)
  return output

def test_model(model: AnomalibModule) -> any:
  engine = _get_engine()
  
  full_ds = get_configured_ds()
  full_otp = engine.test(model, full_ds)
  
  log_ds = get_configured_ds(SampleType.LOGICAL)
  log_otp = engine.test(model, log_ds)
  
  str_ds = get_configured_ds(SampleType.STRUCTURAL)
  str_otp = engine.test(model, str_ds)
  
  output = {
    "full": full_otp[0],
    "logical": log_otp[0],
    "structural": str_otp[0]
  }
  formatted_output = format_metrics(output)
  write_file(FileType.TEST_DATA, formatted_output)
  return output

def run_model(model: AnomalibModule, image_filepath: str):
  engine = Engine()

  prediction = engine.predict(
    model=model,
    data_path=image_filepath
  )
  image = output_to_marked_pil(prediction[0])
  write_file(FileType.IMAGE, image)

def _get_engine() -> Engine:
  config = ModelConfig()
  engine = Engine(
    default_root_dir=config.model_dir,
    accelerator = "gpu" if config.use_cuda else "cpu",
    devices=1,
    **opt("max_epochs", config.max_epochs)
  )

  return engine
