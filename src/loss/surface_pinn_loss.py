import torch
import torch.nn as nn
from dataclasses import dataclass
import numpy as np

@dataclass
class Metadata:
    """Metadata for the batch.
    """
    lat: torch.Tensor
    lon: torch.Tensor
    atmos_levels: tuple[int | float, ...]

@dataclass
class Batch:
    """A batch of data.

    Args:
        surf_vars (dict[str, :class:`torch.Tensor`]): Surface-level variables with shape
            `(b, t, h, w)`.
        static_vars (dict[str, :class:`torch.Tensor`]): Static variables with shape `(h, w)`.
        atmos_vars (dict[str, :class:`torch.Tensor`]): Atmospheric variables with shape
            `(b, t, c, h, w)`.
    """

    surf_vars: dict[str, torch.Tensor]
    static_vars: dict[str, torch.Tensor]
    atmos_vars: dict[str, torch.Tensor]
    metadata: Metadata

class SurfacePressurePhysicallyInformedLoss(nn.Module):
    def __init__(self, 
                 constants=None, 
                 variable_mapping=None,
                 device="cpu"):
        super().__init__()
        
        self.device = device
        
        self.constants = {
            'R': 287.05,
            'cp': 1004.0,
            'kappa': 287.05 / 1004.0,
            'g': 9.80665,
            'R_earth': 6371000,
        }
        
        if constants is not None:
            self.constants.update(constants)
            
        self.variable_mapping = {
            'u': {'container': 'atmos_vars', 'key': 'u'},
            'v': {'container': 'atmos_vars', 'key': 'v'},
            'sp': {'container': 'surf_vars', 'key': 'sp'},
            'msl': {'container': 'surf_vars', 'key': 'msl'},
        }
        
        if variable_mapping is not None:
            self.variable_mapping.update(variable_mapping)
    
    def get_variable(self, batch: Batch, var_name: str):
        if var_name not in self.variable_mapping:
            return None
            
        container_name = self.variable_mapping[var_name]['container']
        key = self.variable_mapping[var_name]['key']
        
        container = getattr(batch, container_name)
        if key in container:
            return container[key].to(self.device)
        return None
    
    def calculate_distance_differentials(self, latitude: torch.Tensor):
        latitude = latitude.to(self.device)
        lat_rad = latitude * torch.pi / 180.0
        
        meters_per_lat = (torch.pi * self.constants['R_earth'] / 180.0) * torch.ones_like(lat_rad, device=self.device)
        meters_per_lon = (torch.pi * self.constants['R_earth'] / 180.0) * torch.cos(lat_rad)
        
        return meters_per_lat.to(self.device), meters_per_lon.to(self.device)
    
    def take_time_derivative(self, field, dt):
        field = field.to(self.device)
        
        der_t = torch.zeros_like(field, device=self.device)
            
        der_t[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dt)
        der_t[:, 0] = (field[:, 1] - field[:, 0]) / dt
        der_t[:, -1] = (field[:, -1] - field[:, -2]) / dt
        
        return der_t.to(self.device)
    
    def take_space_derivative(self, x: torch.Tensor, latitude: torch.Tensor, dx, dy):
        x = x.to(self.device)
        latitude = latitude.to(self.device)
        
        meters_per_lat, meters_per_lon = self.calculate_distance_differentials(latitude)
        
        der_lat = torch.zeros_like(x, device=self.device)
        der_lon = torch.zeros_like(x, device=self.device)
        
        if x.dim() == 5:
            reshape_dims = (1, 1, 1, -1, 1)
        elif x.dim() == 4:
            reshape_dims = (1, 1, -1, 1)
        
        meters_per_lat_reshaped = meters_per_lat.view(*reshape_dims)
        meters_per_lon_reshaped = meters_per_lon.view(*reshape_dims)
        
        actual_dy = dy * meters_per_lat_reshaped
        actual_dx = dx * meters_per_lon_reshaped

        der_lat[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / (2.0 * actual_dy[..., 1:-1, :])
        der_lat[..., 0, :] = (x[..., 1, :] - x[..., 0, :]) / actual_dy[..., 0:1, :]
        der_lat[..., -1, :] = (x[..., -1, :] - x[..., -2, :]) / actual_dy[..., -1:, :]

        der_lon[..., :, 1:-1] = (x[..., :, 2:] - x[..., :, :-2]) / (2.0 * actual_dx[..., :, 0:1])
        der_lon[..., :, 0:1] = (x[..., :, 1:2] - x[..., :, 0:1]) / actual_dx[..., :, 0:1]
        der_lon[..., :, -1:] = (x[..., :, -1:] - x[..., :, -2:-1]) / actual_dx[..., :, 0:1]
        
        return der_lat.to(self.device), der_lon.to(self.device)
    
    def calculate_horizontal_divergence(self, u, v, latitude, dx, dy):
        u = u.to(self.device)
        v = v.to(self.device)
        latitude = latitude.to(self.device)
        
        _, du_dx = self.take_space_derivative(u, latitude, dx, dy)
        dv_dy, _ = self.take_space_derivative(v, latitude, dx, dy)
        
        divergence = du_dx + dv_dy
        
        return divergence.to(self.device)
    
    def calculate_log_surface_pressure_gradient(self, sp, latitude, dx, dy):
        sp = sp.to(self.device)
        latitude = latitude.to(self.device)

        log_sp = torch.log(sp)

        dlog_sp_dy, dlog_sp_dx = self.take_space_derivative(log_sp, latitude, dx, dy)
        
        return dlog_sp_dy, dlog_sp_dx
    
    def calculate_log_surface_pressure_advection(self, u, v, sp, latitude, dx, dy):
        u = u.to(self.device)
        v = v.to(self.device)
        sp = sp.to(self.device)
        latitude = latitude.to(self.device)

        dlog_sp_dy, dlog_sp_dx = self.calculate_log_surface_pressure_gradient(sp, latitude, dx, dy)

        if u.dim() == 5:
            dlog_sp_dx = dlog_sp_dx.unsqueeze(2).expand_as(u)
            dlog_sp_dy = dlog_sp_dy.unsqueeze(2).expand_as(u)

        advection = u * dlog_sp_dx + v * dlog_sp_dy
        
        return advection.to(self.device)
    
    def calculate_sigma_layer_thickness(self, pressure_levels, sp):
        pressure_levels = pressure_levels.to(self.device)
        sp = sp.to(self.device)
        
        b, t, h, w = sp.shape
        levels = len(pressure_levels)

        p_levels = (pressure_levels * 100.0).reshape(1, 1, -1, 1, 1).expand(b, t, -1, h, w)

        sp_expanded = sp.unsqueeze(2).expand(b, t, levels, h, w)

        sigma = p_levels / sp_expanded

        delta_sigma = torch.zeros_like(sigma, device=self.device)

        for k in range(1, levels-1):
            delta_sigma[:, :, k, :, :] = (sigma[:, :, k+1, :, :] - sigma[:, :, k-1, :, :]) / 2.0

        delta_sigma[:, :, 0, :, :] = sigma[:, :, 1, :, :] - sigma[:, :, 0, :, :]
        delta_sigma[:, :, -1, :, :] = sigma[:, :, -1, :, :] - sigma[:, :, -2, :, :]
        
        return delta_sigma.to(self.device)
    
    def vertical_integration(self, field, delta_sigma):
        field = field.to(self.device)
        delta_sigma = delta_sigma.to(self.device)

        integrated_field = torch.sum(field * delta_sigma, dim=2)
        
        return integrated_field.to(self.device)
    
    def calculate_loss(self, batch, dt, dx, dy):
        u = self.get_variable(batch, 'u')
        v = self.get_variable(batch, 'v')
        sp = self.get_variable(batch, 'sp')

        log_sp = torch.log(sp)

        dlog_sp_dt = self.take_time_derivative(log_sp, dt)

        divergence = self.calculate_horizontal_divergence(u, v, batch.metadata.lat, dx, dy)

        advection = self.calculate_log_surface_pressure_advection(u, v, sp, batch.metadata.lat, dx, dy)

        total_effect = divergence + advection

        delta_sigma = self.calculate_sigma_layer_thickness(batch.metadata.atmos_levels, sp)

        integrated_effect = self.vertical_integration(total_effect, delta_sigma)

        rhs = -integrated_effect

        residual = dlog_sp_dt - rhs
        
        return residual.to(self.device)
    
    def forward(self, preds: Batch, target: Batch, dt, dx, dy) -> torch.Tensor:
        preds.metadata.lat = preds.metadata.lat.to(self.device)
        preds.metadata.lon = preds.metadata.lon.to(self.device)
        preds.metadata.atmos_levels = preds.metadata.atmos_levels.to(self.device)
        
        target.metadata.lat = target.metadata.lat.to(self.device)
        target.metadata.lon = target.metadata.lon.to(self.device)
        target.metadata.atmos_levels = target.metadata.atmos_levels.to(self.device)

        era5_residual = self.calculate_loss(target, dt, dx, dy)
        model_residual = self.calculate_loss(preds, dt, dx, dy)

        residual_difference = model_residual - era5_residual
        
        return torch.mean(torch.square(residual_difference)).to(self.device)
    
    def calculate_residuals_from_tensor(self, data_tensor, dt=6, dx=0.25, dy=0.25):
        data_tensor.to(self.device)

        lat = torch.linspace(90, -90, 721).to(self.device)
        lon = torch.linspace(0, 360, 1440+1)[:-1].to(self.device)
        atmos_levels = torch.tensor([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]).to(self.device)

        b, time_steps, _, h, w = data_tensor.shape

        metadata = Metadata(
            lat=lat,
            lon=lon,
            atmos_levels=atmos_levels
        )

        t = data_tensor[:, :, 13:26]
        u = data_tensor[:, :, 26:39]
        v = data_tensor[:, :, 39:52]
        msl = data_tensor[:, :, 68:69]

        original_mapping = self.variable_mapping.copy()
        
        self.variable_mapping = {
            'u': {'container': 'atmos_vars', 'key': 'u'},
            'v': {'container': 'atmos_vars', 'key': 'v'},
            'temperature': {'container': 'atmos_vars', 'key': 't'},
            'sp': {'container': 'surf_vars', 'key': 'msl'},
        }
        
        full_batch = Batch(
            surf_vars={"msl": msl.reshape(b, time_steps, h, w)},
            static_vars={},
            atmos_vars={
                "t": t.reshape(b, time_steps, 13, h, w),
                "u": u.reshape(b, time_steps, 13, h, w),
                "v": v.reshape(b, time_steps, 13, h, w)
            },
            metadata=metadata
        )
        
        residuals = self.calculate_loss(full_batch, dt, dx, dy)
            
        self.variable_mapping = original_mapping
        
        return residuals
