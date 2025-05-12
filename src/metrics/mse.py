from torch import Tensor
import torch

from src.metrics.base_metric import BaseMetric


class MseMetric(BaseMetric):
    def __call__(self, preds: Tensor, ground_truth: Tensor):
        with torch.no_grad():
            return torch.nn.functional.mse_loss(preds.float(), ground_truth.float()).item()
