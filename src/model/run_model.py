from utils.dict import opt
from utils.image import output_to_marked_pil
from ModelConfig import ModelConfig, ModelType
from utils.io import write_file, move_model, FileType
from metrics.metrics import format_metrics, calculate_metrics
from dataset.SubMVTecLOCO import get_configured_ds, SampleType, SubMVTecLOCO

from anomalib.engine import Engine
from anomalib.models import AnomalibModule

from .dual_patchcore.lightning_model import DualPatchcore

def train_model(model: AnomalibModule) -> any:
  dataset = get_configured_ds()
  engine = _get_engine()
  output = engine.fit(model, dataset)
  move_model(engine.trainer.checkpoint_callback.best_model_path)
  return output

def train_dual_patchcore(model: DualPatchcore) -> None:
  engine = _get_engine()
  dataset = get_configured_ds(use_default=True)
  engine.fit(model, dataset)
  model.initiate_secondary_training()
  dataset = get_configured_ds()
  engine.fit(model, dataset)
  move_model(engine.trainer.checkpoint_callback.best_model_path)

def test_model(model: AnomalibModule) -> any:
  full_ds = get_configured_ds()
  full_otp = _test_model(model, full_ds)
  
  log_ds = get_configured_ds(SampleType.LOGICAL)
  log_otp = _test_model(model, log_ds)
  
  str_ds = get_configured_ds(SampleType.STRUCTURAL)
  str_otp = _test_model(model, str_ds)
  
  output = {
    "full": full_otp[0],
    "logical": log_otp[0],
    "structural": str_otp[0]
  }
  formatted_output = format_metrics(output)
  write_file(FileType.TEST_DATA, formatted_output)
  return output

# Anomalib's metric calculation with DualPatchcore fails, thus this method is introduced
# Metrics are still held within the model configuration, because they were already created using such configuration
def _test_model(model: AnomalibModule, dataset: SubMVTecLOCO) -> any:
  engine = _get_engine()
  predictions = engine.predict(model, datamodule=dataset)
  return calculate_metrics(predictions)

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
