from ModelConfig import ModelConfig
from anomalib.data import MVTecLOCO

from pathlib import Path
from pandas import DataFrame
from torchvision.transforms.v2 import Transform
from anomalib.data.utils import TestSplitMode, ValSplitMode

class SubMVTecLOCO(MVTecLOCO):
  def __init__(
    self,
    root: Path | str | None,
    category: str,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 8,
    train_augmentations: Transform | None = None,
    val_augmentations: Transform | None = None,
    test_augmentations: Transform | None = None,
    augmentations: Transform | None = None,
    test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
    val_split_mode: ValSplitMode | str = ValSplitMode.FROM_DIR,
    test_split_ratio: float | None = None,
    val_split_ratio: float | None = None,
    seed: int | None = None,
  ) -> None:
    super().__init__(
      root=root or "./datasets/MVTec_LOCO",
      category=category or "breakfast_box",
      train_batch_size=train_batch_size,
      eval_batch_size=eval_batch_size,
      num_workers=num_workers,
      train_augmentations=train_augmentations,
      val_augmentations=val_augmentations,
      test_augmentations=test_augmentations,
      augmentations=augmentations,
      test_split_mode=test_split_mode,
      val_split_mode=val_split_mode,
      test_split_ratio=test_split_ratio,
      val_split_ratio=val_split_ratio,
      seed=seed
    )

  def _setup(self, _stage: str | None = None) -> None:
    super()._setup(_stage)
    config = ModelConfig()

    if config.img_count != None:
      self.train_data.samples = _reduce_dataset(self.train_data.samples, config.img_count, config.random_samples)
      
def get_configured_ds() -> SubMVTecLOCO:
  config = ModelConfig()
  dataset = SubMVTecLOCO(
    root=config.ds_path,
    category=config.category,
    num_workers=config.worker_count,
    train_batch_size=config.batch_size,
    eval_batch_size=config.batch_size,
  )
  dataset.setup()

  return dataset

def _reduce_dataset(dataset: DataFrame, size: int = None, random_samples: bool = False) -> DataFrame:
  if size == None:
    return
  if size < 1:
    raise ValueError("Train dataset must have a positive number of data")
  
  if size > len(dataset):
    raise ValueError(f"Train dataset contains {len(dataset)} images, but trying to generate split with {size} images")

  if random_samples:
    dataset = dataset.sample(n=size)
  else:
    dataset = dataset[:size]
  
  return dataset