from anomalib.metrics.evaluator import Evaluator
from torchmetrics.classification import BinaryJaccardIndex
from anomalib.metrics import AUROC, F1Score, AUPR, AnomalibMetric

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

def calculate_metrics(model_output: any) -> dict:
  auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
  f1 = F1Score(fields=["pred_score", "gt_label"], prefix="image_")
  aupr = AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
  iou = IoU(fields=["pred_mask", "gt_mask"], prefix="pixel_")

  for batch in model_output:
    auroc.update(batch)
    f1.update(batch)
    aupr.update(batch)
    iou.update(batch)

  return {
    'image_AUROC': auroc.compute(),
    'image_F1Score': f1.compute(),
    'pixel_AUPR': aupr.compute(),
    'pixel_IoU': iou.compute(),
  }  

def _format_set(metrics: dict) -> str:
  return f"   {metrics['image_AUROC']:5.4f}  | {metrics['image_F1Score']:5.4f} |   {metrics['pixel_AUPR']:5.4f}   |  {metrics['pixel_IoU']:5.4f}\n"

class IoU(AnomalibMetric, BinaryJaccardIndex):
  pass 