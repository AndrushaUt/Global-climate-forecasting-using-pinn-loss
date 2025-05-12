import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# import fsspec
import gcsfs
# import pandas as pd
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

data = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr')
for key in KEYS:
    print(f"{key} - {data[key].shape}")
    print(type(data[key]))

print(type(torch.from_numpy(data['specific_humidity'].values)))
# a = torch.tensor([1,2,3,4,5,6], dtype=float)
# print(torch.mean(torch.square(a)))