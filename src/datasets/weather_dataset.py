from src.datasets.base_dataset import BaseDataset
import numpy as np
import torch
import multiprocessing as mp
from tqdm import tqdm
import os
import time


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
        dataset_dir,
        data_start = None,
        data_end = None
    ) -> None:
        data_start = data_start
        data_end = data_end

        data = {
            "2mt": os.path.join(dataset_dir, "2m_temperature_data"),
            "10u": os.path.join(dataset_dir, "10m_u_component_of_wind_data"),
            "10v": os.path.join(dataset_dir, "10m_v_component_of_wind_data"),
            "msl": os.path.join(dataset_dir, "mean_sea_level_pressure_data"),
            "t": os.path.join(dataset_dir, "temperature_data"),
            "z": os.path.join(dataset_dir, "geopotential_data"),
            "v": os.path.join(dataset_dir, "v_component_of_wind_data"),
            "u": os.path.join(dataset_dir, "u_component_of_wind_data"),
            "q": os.path.join(dataset_dir, "specific_humidity_data")
        }

        super().__init__(index=data)

