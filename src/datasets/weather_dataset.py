from src.datasets.base_dataset import BaseDataset
import numpy as np
import torch

KEYS = {
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "geopotential",
        "v_component_of_wind",
        "u_component_of_wind",
        "specific_humidity",
    }


class WeatherDataset(BaseDataset):
    def __init__(
        self,
        two_m_temperature_path,
        ten_m_u_component_of_wind_path,
        ten_m_v_component_of_wind_path,
        mean_sea_level_pressure_path,
        geopotential_path,
        v_component_of_wind_path,
        u_component_of_wind_path,
        specific_humidity_path,
        date_start = None,
        data_end = None
    ) -> None:
        two_m_temperature = torch.from_numpy(np.load(two_m_temperature_path))
        ten_m_u_component_of_wind = torch.from_numpy(np.load(ten_m_u_component_of_wind_path))
        ten_m_v_component_of_wind = torch.from_numpy(np.load(ten_m_v_component_of_wind_path))
        mean_sea_level_pressure = torch.from_numpy(np.load(mean_sea_level_pressure_path))
        geopotential = torch.from_numpy(np.load(geopotential_path))
        v_component_of_wind = torch.from_numpy(np.load(v_component_of_wind_path))
        u_component_of_wind = torch.from_numpy(np.load(u_component_of_wind_path))
        specific_humidity = torch.from_numpy(np.load(specific_humidity_path))

        data_start = date_start or 0
        data_end = data_end or two_m_temperature.shape[0]
        two_m_temperature = two_m_temperature[data_start:data_end, :, :]
        ten_m_u_component_of_wind = ten_m_u_component_of_wind[data_start:data_end, :, :]
        ten_m_v_component_of_wind = ten_m_v_component_of_wind[data_start:data_end, :, :]
        mean_sea_level_pressure = mean_sea_level_pressure[data_start:data_end, :, :]
        geopotential = geopotential[data_start:data_end, :, :, :]
        v_component_of_wind = v_component_of_wind[data_start:data_end, :, :, :]
        u_component_of_wind = u_component_of_wind[data_start:data_end, :, :, :]
        specific_humidity = specific_humidity[data_start:data_end, :, :, :]

        data = {
            "2mt": two_m_temperature,
            "10u": ten_m_u_component_of_wind,
            "10v": ten_m_v_component_of_wind,
            "msl": mean_sea_level_pressure,
            "z": geopotential,
            "v": v_component_of_wind,
            "u": u_component_of_wind,
            "q": specific_humidity
        }

        super().__init__(index=data)

