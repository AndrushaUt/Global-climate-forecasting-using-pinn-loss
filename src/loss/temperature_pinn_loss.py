import torch
import torch.nn as nn
from dataclasses import dataclass

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

class TemperaturePhysicallyInformedLoss(nn.Module):
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
        }
        
        if constants is not None:
            self.constants.update(constants)
            
        self.variable_mapping = {
            'u': {'container': 'atmos_vars', 'key': 'u'},
            'v': {'container': 'atmos_vars', 'key': 'v'},
            'temperature': {'container': 'atmos_vars', 'key': 't'},
            'specific_humidity': {'container': 'atmos_vars', 'key': 'q'},
            'geopotential': {'container': 'atmos_vars', 'key': 'z'},
            'msl': {'container': 'surf_vars', 'key': 'msl'},
        }
        
        if variable_mapping is not None:
            self.variable_mapping.update(variable_mapping)
    
    def get_variable(self,
            batch: Batch,
            var_name: str
        ):
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
        
        meters_per_lat = (torch.pi * self.constants['R'] / 180.0) * torch.ones_like(lat_rad)
        meters_per_lon = (torch.pi * self.constants['R'] / 180.0) * torch.cos(lat_rad)
        
        return meters_per_lat, meters_per_lon
    
    def take_time_derivative(self, field, dt):
        field = field.to(self.device)
        
        der_t = torch.zeros_like(field)
            
        der_t[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dt)
        der_t[:, 0] = (field[:, 1] - field[:, 0]) / dt
        der_t[:, -1] = (field[:, -1] - field[:, -2]) / dt
        
        return der_t
    
    def take_space_derivative(self, x: torch.Tensor, latitude: torch.Tensor, dx, dy):
        x = x.to(self.device)
        latitude = latitude.to(self.device)
        
        meters_per_lat, meters_per_lon = self.calculate_distance_differentials(latitude)
        
        der_lat = torch.zeros_like(x)
        der_lon = torch.zeros_like(x)
        
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
        
        return der_lat, der_lon

    def calculate_sigma_dot(self, pressure_levels, msl, dt):
        pressure_levels = pressure_levels.to(self.device)
        msl = msl.to(self.device)
        
        b, t, h, w = msl.shape
        pressure_levels = (pressure_levels * 100).reshape(1, 1, -1, 1, 1).repeat(b, t, 1, h, w)
        ps_expanded = msl.unsqueeze(2)
        sigma = pressure_levels / ps_expanded
        
        sigma_dot = self.take_time_derivative(sigma, dt)
        
        return sigma_dot
    
    def calculate_horizontal_temperature_advection(self, u, v, T, latitude, dx, dy):
        u = u.to(self.device)
        v = v.to(self.device)
        T = T.to(self.device)
        
        dT_dy, dT_dx = self.take_space_derivative(T, latitude, dx, dy)

        advection = u * dT_dx + v * dT_dy
        
        return -advection
    
    def calculate_exact_sigma_derivatives(self, pressure_levels, msl):
        pressure_levels = pressure_levels.to(self.device)
        msl = msl.to(self.device)

        b, t, h, w = msl.shape
        p_levels = (pressure_levels * 100).reshape(1, 1, -1, 1, 1).repeat(b, t, 1, h, w)
        ps_expanded = msl.unsqueeze(2)
        sigma = p_levels / ps_expanded

        delta_sigma = torch.abs(sigma[:, :, 1:, ...] - sigma[:, :, :-1, ...])
        
        return delta_sigma

    def take_vertical_derivative_sigma(self, x, delta_sigma):
        x = x.to(self.device)
        der_sigma = torch.zeros_like(x)
        n_levels = x.shape[2]

        for i in range(1, n_levels-1):
            delta_up = delta_sigma[:, :, i, ...]
            delta_down = delta_sigma[:, :, i-1, ...]

            weight_up = delta_down / (delta_up + delta_down)
            weight_down = delta_up / (delta_up + delta_down)
            
            der_sigma[:, :, i, ...] = (
                weight_up * (x[:, :, i+1, ...] - x[:, :, i, ...]) / delta_up +
                weight_down * (x[:, :, i, ...] - x[:, :, i-1, ...]) / delta_down
            )

        der_sigma[:, :, 0, ...] = (x[:, :, 1, ...] - x[:, :, 0, ...]) / delta_sigma[:, :, 0, ...]
        der_sigma[:, :, -1, ...] = (x[:, :, -1, ...] - x[:, :, -2, ...]) / delta_sigma[:, :, -2, ...]
        
        return der_sigma.to(self.device)

    def calculate_vertical_temperature_advection(self, T, sigma_dot, pressure_levels, msl):
        T = T.to(self.device)
        sigma_dot = sigma_dot.to(self.device)
        delta_sigma = self.calculate_exact_sigma_derivatives(pressure_levels, msl)

        dT_dsigma = self.take_vertical_derivative_sigma(T, delta_sigma)

        advection = -sigma_dot * dT_dsigma
        
        return advection
    
    def calculate_omega(self, sigma_dot, pressure_levels, msl):
        sigma_dot = sigma_dot.to(self.device)
        pressure_levels = pressure_levels.to(self.device)
        msl = msl.to(self.device)

        b, t, h, w = msl.shape
        p_levels = (pressure_levels * 100).reshape(1, 1, -1, 1, 1).repeat(b, t, 1, h, w)

        ps_expanded = msl.unsqueeze(2)

        p = p_levels * ps_expanded / p_levels[:,:,-1:,:,:]

        omega = sigma_dot * p
        
        return omega
    
    def calculate_adiabatic_term(self, T, omega, pressure_levels, msl):
        T = T.to(self.device)
        omega = omega.to(self.device)
        pressure_levels = pressure_levels.to(self.device)
        msl = msl.to(self.device)

        b, t, h, w = msl.shape
        p_levels = (pressure_levels * 100).reshape(1, 1, -1, 1, 1).repeat(b, t, 1, h, w)
    
        ps_expanded = msl.unsqueeze(2)

        p = p_levels * ps_expanded / p_levels[:,:,-1:,:,:]

        adiabatic = self.constants['kappa'] * T * omega / p
        
        return adiabatic
    
    def calculate_temperature_tendency(self, batch, dt, dx, dy, dz=None):
        u = self.get_variable(batch, 'u')
        v = self.get_variable(batch, 'v')
        T = self.get_variable(batch, 'temperature')
        msl = self.get_variable(batch, 'msl')

        dT_dt = self.take_time_derivative(T, dt)

        sigma_dot = self.calculate_sigma_dot(batch.metadata.atmos_levels, msl, dt)

        omega = self.calculate_omega(sigma_dot, batch.metadata.atmos_levels, msl)

        h_advection = self.calculate_horizontal_temperature_advection(u, v, T, batch.metadata.lat, dx, dy)

        v_advection = self.calculate_vertical_temperature_advection(T, sigma_dot, batch.metadata.atmos_levels, msl)

        adiabatic = self.calculate_adiabatic_term(T, omega, batch.metadata.atmos_levels, msl)

        rhs = h_advection + v_advection + adiabatic

        residual = dT_dt - rhs
        
        return residual
    
    def forward(self, 
                preds: Batch, 
                target: Batch, 
                dt, 
                dx, 
                dy) -> torch.Tensor:
        preds.metadata.lat = preds.metadata.lat.to(self.device)
        preds.metadata.lon = preds.metadata.lon.to(self.device)
        preds.metadata.atmos_levels = preds.metadata.atmos_levels.to(self.device)
        
        target.metadata.lat = target.metadata.lat.to(self.device)
        target.metadata.lon = target.metadata.lon.to(self.device)
        target.metadata.atmos_levels = target.metadata.atmos_levels.to(self.device)

        era5_residual = self.calculate_temperature_tendency(target, dt, dx, dy)
        model_residual = self.calculate_temperature_tendency(preds, dt, dx, dy)

        residual_difference = model_residual - era5_residual

        return torch.mean(torch.square(residual_difference))

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
            'msl': {'container': 'surf_vars', 'key': 'msl'},
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
        
        residuals = self.calculate_temperature_tendency(full_batch, dt, dx, dy)
            
        self.variable_mapping = original_mapping
        
        return residuals
