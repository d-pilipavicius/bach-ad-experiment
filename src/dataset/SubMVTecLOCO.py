from enum import Enum

from utils.dict import opt
from ModelConfig import ModelConfig
from anomalib.data import MVTecLOCO

from pathlib import Path
from pandas import DataFrame, concat
from torchvision.transforms.v2 import Transform
from anomalib.data.utils import TestSplitMode, ValSplitMode

class SampleType(Enum):
  GOOD = "good"
  LOGICAL = "logical_anomalies"
  STRUCTURAL = "structural_anomalies"

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
    test_split_ratio: float | None = 0,
    val_split_ratio: float | None = None,
    seed: int | None = None,
    use_test_split: SampleType | None = None,
  ) -> None:
    super().__init__(
      **opt("root", root),
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
    self.use_test_split = use_test_split

  def _setup(self, _stage: str | None = None) -> None:
    super()._setup(_stage)
    config = ModelConfig()

    train_ds = self.train_data.samples
    test_ds = self.test_data.samples

    if config.img_trn_count is not None:
      train_ds = _reduce_dataset(train_ds, config.img_trn_count, config.random_trn_samples)

    if config.trn_logical is not None:
      test_ds, split = _split_dataset(test_ds, SampleType.LOGICAL, config.trn_logical)
      train_ds = concat([train_ds, split], ignore_index=True)

    if config.trn_structural is not None:
      test_ds, split = _split_dataset(test_ds, SampleType.STRUCTURAL, config.trn_structural)
      train_ds = concat([train_ds, split], ignore_index=True)

    if self.use_test_split is not None:
      test_ds = _select_test_data(test_ds, self.use_test_split)

    self.train_data.samples = train_ds
    self.test_data.samples = test_ds

def get_configured_ds(test_split: SampleType | None = None) -> SubMVTecLOCO:
  config = ModelConfig()
  dataset = SubMVTecLOCO(
    root=config.ds_path,
    category=config.category,
    num_workers=config.worker_count,
    **opt("train_batch_size",config.batch_size),
    **opt("eval_batch_size",config.batch_size),
    **opt("use_test_split", test_split),
  )
  dataset.setup()

  return dataset

def _reduce_dataset(dataset: DataFrame, size: int = None, random_samples: bool | None = False) -> DataFrame:
  if random_samples is None:
    random_samples = False  
  if size is None:
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

def _select_test_data(dataset: DataFrame, sample_type: SampleType) -> DataFrame:
  return dataset[dataset["label"] == sample_type.value]

def _split_dataset(dataset: DataFrame, sample_type: SampleType, sample_names: list[str]) -> tuple[DataFrame, DataFrame]:
  subset = dataset[
    (dataset["label"] == sample_type.value) &
    (dataset["image_path"].apply(lambda x: any(name in x for name in sample_names)))
  ]
  remaining = dataset.drop(subset.index)
  return remaining, subset