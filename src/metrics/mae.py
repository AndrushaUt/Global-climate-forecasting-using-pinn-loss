from torch import Tensor
import torch

from src.metrics.base_metric import BaseMetric


class MaeMetric(BaseMetric):
    def __call__(self, preds: Tensor, ground_truth: Tensor):
        with torch.no_grad():
            return torch.nn.functional.l1_loss(preds.float(), ground_truth.float(), reduction='mean').item()
