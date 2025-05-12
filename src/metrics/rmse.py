from torch import Tensor
import torch

from src.metrics.base_metric import BaseMetric


class RmseMetric(BaseMetric):
    def __call__(self, preds: Tensor, ground_truth: Tensor):
        with torch.no_grad():
            return torch.sqrt(torch.mean((preds.float() - ground_truth.float()) ** 2)).item()
