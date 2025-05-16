import torch.nn as nn
from src.loss import TemperaturePhysicallyInformedLoss, PhysicallyInformedLoss, HumidityPhysicallyInformedLoss
import torch


class CombinedLoss(nn.Module):
    def __init__(self, physical_weight=1, use_mae=False, device="cuda"):
        super().__init__()
        self.physical_weight = physical_weight
        self.use_mae = use_mae
        self.device = device
    
    def forward(self, pred_tensor, target_tensor, batch, dt=6, dx=5.625, dy=5.625):
        physical_loss = HumidityPhysicallyInformedLoss(device=self.device)
        pred_residuals = physical_loss.calculate_residuals_from_tensor(pred_tensor.permute(0, 1, 4, 2, 3), dt, dx, dy)
        target_residuals = physical_loss.calculate_residuals_from_tensor(target_tensor.permute(0, 1, 4, 2, 3), dt, dx, dy)

        pin_loss_value = torch.mean(torch.abs(pred_residuals - target_residuals))

        if self.use_mae:
            direct_loss = nn.functional.l1_loss(pred_tensor, target_tensor)
        else:
            direct_loss = nn.functional.mse_loss(pred_tensor, target_tensor)

        total_loss = self.physical_weight * pin_loss_value + (1 - self.physical_weight) * direct_loss

        return {"loss": total_loss}
    
    def _denormalize_data(self, data, channel_mapping, norm_params):
        for key, val in channel_mapping['key_to_channels'].items():
            if len(val) == 0:
                mean = norm_params[key]['mean']
                std = norm_params[key]['std']
                data[:, :, val[0]] = data[:, :, val[0]] * std + mean
                norm_params[key] = {"mean": mean, "std": std}
            else:
                for level in val:
                    mean = norm_params[key]['mean'][level - min(val)]
                    std = norm_params[key]['std'][level - min(val)]
                    data[:, :, level] = data[:, :, level]*std + mean
        
        return data
        