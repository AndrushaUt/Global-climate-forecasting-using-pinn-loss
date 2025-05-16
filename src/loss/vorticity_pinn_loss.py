import torch
import torch.nn as nn
from dataclasses import dataclass

# TODO: change to Batch class from aurora


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


class PhysicallyInformedLoss(nn.Module):
    def __init__(self, 
                 constants=None, 
                 variable_mapping=None,
                 device="cpu"):
        super().__init__()
        
        self.device = device
        
        self.constants = {
            'R': 287.05,
            'omega': 7.292e-5,
            'R_earth': 6371000.0,
        }
        
        if constants is not None:
            self.constants.update(constants)
            
        self.variable_mapping = {
            'u': {'container': 'atmos_vars', 'key': 'u'},
            'v': {'container': 'atmos_vars', 'key': 'v'},
            'temperature': {'container': 'atmos_vars', 'key': 't'},
            'surface_pressure': {'container': 'surf_vars', 'key': 'msl'},
        }
        
        if variable_mapping is not None:
            self.variable_mapping.update(variable_mapping)
            
        self.epsilon = 1e-8
    
    def get_variable(self, batch, var_name):
        if var_name not in self.variable_mapping:
            return None
            
        container_name = self.variable_mapping[var_name]['container']
        key = self.variable_mapping[var_name]['key']
        
        container = getattr(batch, container_name)
        if key in container:
            return container[key].to(self.device)
        return None
    
    def calculate_distance_differentials(self, latitude):
        latitude = latitude.to(self.device)
        lat_rad = latitude * torch.pi / 180.0
        
        meters_per_lat = (torch.pi * self.constants['R_earth'] / 180.0) * torch.ones_like(lat_rad)
        meters_per_lon = (torch.pi * self.constants['R_earth'] / 180.0) * torch.cos(lat_rad)
        
        return meters_per_lat, meters_per_lon
    
    def take_time_derivative(self, field, dt):
        field = field.to(self.device)
        dt = dt.to(self.device) if isinstance(dt, torch.Tensor) else dt
        
        der_t = torch.zeros_like(field)
            
        der_t[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dt + self.epsilon)
        der_t[:, 0] = (field[:, 1] - field[:, 0]) / (dt + self.epsilon)
        der_t[:, -1] = (field[:, -1] - field[:, -2]) / (dt + self.epsilon)
        
        return der_t
    
    def take_space_derivative(self, x, latitude, dx, dy):
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
        
        der_lat[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / (2.0 * actual_dy[..., 1:-1, :] + self.epsilon)
        der_lat[..., 0, :] = (x[..., 1, :] - x[..., 0, :]) / (actual_dy[..., 0:1, :] + self.epsilon)
        der_lat[..., -1, :] = (x[..., -1, :] - x[..., -2, :]) / (actual_dy[..., -1:, :] + self.epsilon)
    
        der_lon[..., :, 1:-1] = (x[..., :, 2:] - x[..., :, :-2]) / (2.0 * actual_dx[..., :, 0:1] + self.epsilon)
        der_lon[..., :, 0:1] = (x[..., :, 1:2] - x[..., :, 0:1]) / (actual_dx[..., :, 0:1] + self.epsilon)
        der_lon[..., :, -1:] = (x[..., :, -1:] - x[..., :, -2:-1]) / (actual_dx[..., :, 0:1] + self.epsilon)
        
        return der_lat, der_lon

    def calculate_level_distances(self, pressure_levels, surface_pressure):
        """
        Вычисляет точные расстояния между соседними сигма-уровнями.
        """
        pressure_levels = pressure_levels.to(self.device)
        surface_pressure = surface_pressure.to(self.device)
    
        b, t, h, w = surface_pressure.shape
        p_levels = (pressure_levels * 100).reshape(1, 1, -1, 1, 1).repeat(b, t, 1, h, w)
        ps_expanded = surface_pressure.unsqueeze(2)
        
        sigma = p_levels / (ps_expanded + self.epsilon)

        distances = torch.abs(sigma[:, :, 1:, ...] - sigma[:, :, :-1, ...])
        
        distances = torch.clamp(distances, min=self.epsilon)
        
        return distances, sigma

    def take_vertical_derivative_with_distances(self, x, distances):
        """
        Вычисляет вертикальную производную с использованием точных расстояний.
        """
        x = x.to(self.device)
        distances = distances.to(self.device)
        der_z = torch.zeros_like(x)
        n_levels = x.shape[2]

        for i in range(1, n_levels-1):
            delta_up = distances[:, :, i, ...]
            delta_down = distances[:, :, i-1, ...]

            der_z[:, :, i, ...] = (
                x[:, :, i+1, ...] - x[:, :, i-1, ...]
            ) / (delta_up + delta_down + self.epsilon)

        der_z[:, :, 0, ...] = (x[:, :, 1, ...] - x[:, :, 0, ...]) / (distances[:, :, 0, ...] + self.epsilon)
        der_z[:, :, -1, ...] = (x[:, :, -1, ...] - x[:, :, -2, ...]) / (distances[:, :, -2, ...] + self.epsilon)
        
        return der_z

    def take_vertical_derivative(self, x, h_z=1.0):
        x = x.to(self.device)
        h_z = h_z.to(self.device) if isinstance(h_z, torch.Tensor) else h_z
        
        der_z = torch.zeros_like(x)
        
        der_z[:, :, 1:-1, ...] = (x[:, :, 2:, ...] - x[:, :, :-2, ...]) / (2.0 * h_z + self.epsilon)
        der_z[:, :, 0, ...] = (x[:, :, 1, ...] - x[:, :, 0, ...]) / (h_z + self.epsilon)
        der_z[:, :, -1, ...] = (x[:, :, -1, ...] - x[:, :, -2, ...]) / (h_z + self.epsilon)
        
        return der_z
    
    def calculate_coriolis_parameter(self, latitude, longitude):
        latitude = latitude.to(self.device)
        longitude = longitude.to(self.device)

        lat_grid, _ = torch.meshgrid(latitude, longitude, indexing='ij')
        lat_rad = lat_grid * torch.pi / 180.0
        f = 2 * self.constants['omega'] * torch.sin(lat_rad)
        
        return f
    
    def calculate_relative_vorticity(self, u, v, latitude, dx, dy):
        u = u.to(self.device)
        v = v.to(self.device)
        latitude = latitude.to(self.device)
        
        du_dy, _ = self.take_space_derivative(u, latitude, dx, dy)
        _, dv_dx = self.take_space_derivative(v, latitude, dx, dy)
        
        vorticity = dv_dx - du_dy
        
        return vorticity
    
    def calculate_sigma_dot(self, pressure_levels, surface_pressure, dt):
        pressure_levels = pressure_levels.to(self.device)
        surface_pressure = surface_pressure.to(self.device)
        
        b,t,h,w = surface_pressure.shape
        pressure_levels = (pressure_levels * 100).reshape(1, 1, -1, 1, 1).repeat(b, t, 1, h, w)
        ps_expanded = surface_pressure.unsqueeze(2)
        
        sigma = pressure_levels / (ps_expanded + self.epsilon)
        
        sigma_dot = self.take_time_derivative(sigma, dt)
        
        return sigma_dot

    def calculate_vector_field(self, u, v, vorticity, f, sigma_dot, T, ps, latitude, dx, dy, pressure_levels=None):
        u = u.to(self.device)
        v = v.to(self.device)
        vorticity = vorticity.to(self.device)
        f = f.to(self.device)
        sigma_dot = sigma_dot.to(self.device)
        T = T.to(self.device)
        ps = ps.to(self.device)
        latitude = latitude.to(self.device)
        
        if f.dim() < vorticity.dim():
            if vorticity.dim() == 5:
                b, t, c, _, _ = vorticity.shape
                f = f.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(b, t, c, -1, -1)
            elif vorticity.dim() == 4:
                b, t, _, _ = vorticity.shape
                f = f.unsqueeze(0).unsqueeze(0).expand(b, t, -1, -1)
        
        total_vorticity = vorticity + f
        
        k_cross_u_x = -v
        k_cross_u_y = u
        k_cross_u_z = torch.zeros_like(u)
        
        term1_x = total_vorticity * k_cross_u_x
        term1_y = total_vorticity * k_cross_u_y
        term1_z = total_vorticity * k_cross_u_z

        if pressure_levels is not None:
            distances, _ = self.calculate_level_distances(pressure_levels, ps)
            du_dsigma = self.take_vertical_derivative_with_distances(u, distances)
            dv_dsigma = self.take_vertical_derivative_with_distances(v, distances)
        else:
            du_dsigma = torch.zeros_like(u)
            dv_dsigma = torch.zeros_like(v)
        
        term2_x = sigma_dot * du_dsigma
        term2_y = sigma_dot * dv_dsigma
        term2_z = torch.zeros_like(u)
        
        log_ps = torch.log(ps + self.epsilon)
        _, dlog_ps_dx = self.take_space_derivative(log_ps, latitude, dx, dy)
        dlog_ps_dy, _ = self.take_space_derivative(log_ps, latitude, dx, dy)
        
        if dlog_ps_dx.dim() < T.dim():
            if T.dim() == 5:
                b, t, c, h, w = T.shape
                dlog_ps_dx = dlog_ps_dx.unsqueeze(2).expand(b, t, c, h, w)
                dlog_ps_dy = dlog_ps_dy.unsqueeze(2).expand(b, t, c, h, w)
        
        R = self.constants['R']
        term3_x = R * T * dlog_ps_dx
        term3_y = R * T * dlog_ps_dy
        term3_z = torch.zeros_like(T)

        vector_field_x = term1_x + term2_x + term3_x
        vector_field_y = term1_y + term2_y + term3_y
        vector_field_z = term1_z + term2_z + term3_z
        
        return vector_field_x, vector_field_y, vector_field_z
    
    def calculate_curl(self, vector_field, latitude, dx, dy):
        u, v, _ = vector_field
        latitude = latitude.to(self.device)
        
        du_dy, _ = self.take_space_derivative(u, latitude, dx, dy)
        _, dv_dx = self.take_space_derivative(v, latitude, dx, dy)
        
        curl_z = dv_dx - du_dy
        
        return curl_z

    def calculate_loss(self, batch, dt, dx, dy):
        batch.metadata.lat = batch.metadata.lat.to(self.device)
        batch.metadata.lon = batch.metadata.lon.to(self.device)
        batch.metadata.atmos_levels = batch.metadata.atmos_levels.to(self.device)
        
        u = self.get_variable(batch, 'u')
        v = self.get_variable(batch, 'v')
        T = self.get_variable(batch, 'temperature')
        ps = self.get_variable(batch, 'surface_pressure')
        
        vorticity = self.get_variable(batch, 'vorticity')
        if vorticity is None:
            vorticity = self.calculate_relative_vorticity(u, v, batch.metadata.lat, dx, dy)
        
        f = self.calculate_coriolis_parameter(batch.metadata.lat, batch.metadata.lon)
        
        dvorticity_dt = self.take_time_derivative(vorticity, dt)

        sigma_dot = self.calculate_sigma_dot(batch.metadata.atmos_levels, ps, dt)

        vector_field = self.calculate_vector_field(
            u, v, vorticity, f, sigma_dot, T, ps, batch.metadata.lat, dx, dy, 
            pressure_levels=batch.metadata.atmos_levels
        )

        curl_term = self.calculate_curl(vector_field, batch.metadata.lat, dx, dy)

        loss = dvorticity_dt + curl_term

        return loss

    def forward(self,
            preds: Batch,
            target: Batch,
            dt,
            dx, 
            dy
        ) -> torch.Tensor:
        """        
        Parameters:
        target : Batch
        preds : Batch
        dt, dx, dy : float
            
        Returns: Tensor (1)
        """
        era5_loss = self.calculate_loss(target, dt, dx, dy)
        model_loss = self.calculate_loss(preds, dt, dx, dy)

        loss_difference = model_loss - era5_loss

        return torch.mean(torch.square(loss_difference))

    def calculate_residuals_from_tensor(self, data_tensor, dt=6, dx=0.25, dy=0.25):
        data_tensor.to(self.device)

        lat = torch.linspace(90, -90, 32).to(self.device)
        lon = torch.linspace(0, 360, 64+1)[:-1].to(self.device)
        atmos_levels = torch.tensor([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]).to(self.device)

        b, time_steps, _, h, w = data_tensor.shape

        metadata = Metadata(
            lat=lat,
            lon=lon,
            atmos_levels=atmos_levels
        )

        t = data_tensor[:, :, 17:30]
        u = data_tensor[:, :, 30:43]
        v = data_tensor[:, :, 43:56]
        msl = data_tensor[:, :, 3:4]

        original_mapping = self.variable_mapping.copy()
        
        self.variable_mapping = {
            'u': {'container': 'atmos_vars', 'key': 'u'},
            'v': {'container': 'atmos_vars', 'key': 'v'},
            'temperature': {'container': 'atmos_vars', 'key': 't'},
            'surface_pressure': {'container': 'surf_vars', 'key': 'msl'},
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
