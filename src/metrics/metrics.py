from anomalib.metrics.evaluator import Evaluator
from anomalib.metrics import AUROC, F1Score, AUPR, F1AdaptiveThreshold, AnomalibMetric
from torchmetrics.classification import BinaryJaccardIndex

def get_evaluator() -> Evaluator:
  return Evaluator(
    test_metrics=[
      AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
      F1Score(fields=["pred_score", "gt_label"], prefix="image_"),
      AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
      IoU(fields=["pred_mask", "gt_mask"], prefix="pixel_")
    ]
  )

def format_metrics(model_output: dict) -> str:
  metrics = "  TEST_DS  | AUROC_img | F1_img | AUPR_pixel | IoU_pixel\n"
  metrics += f"   FULL    |{_format_set(model_output['full'])}"
  metrics += f"  LOGICAL  |{_format_set(model_output['logical'])}"
  metrics += f"STRUCTURAL |{_format_set(model_output['structural'])}"

  return metrics

def _format_set(metrics: dict) -> str:
  return f"   {metrics['image_AUROC']:5.4f}  | {metrics['image_F1Score']:5.4f} |   {metrics['pixel_AUPR']:5.4f}   |  {metrics['pixel_IoU']:5.4f}\n"

class IoU(AnomalibMetric, BinaryJaccardIndex):
  pass 