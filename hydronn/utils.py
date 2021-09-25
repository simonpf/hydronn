"""
=============
hydronn.utils
=============

Various utility functions.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
import xarray as xr

def plot_scenes(dataset, n=3):

    n_scenes = dataset.scenes.size
    indices = np.random.permutation(n_scenes)

    f, axs = plt.subplots(n, 3, figsize=(12, n * 4))

    precip_norm = LogNorm(1e-1, 1e2)

    for i in range(n):
        scene = dataset[{"scenes": indices[i]}]

        x = scene.x_500.data
        y = scene.y_500.data
        z = scene.C02.data
        axs[i, 0].pcolormesh(x, y, z)

        x = scene.x_.data
        y = scene.y_.data
        z = scene.C08.data
        axs[i, 1].pcolormesh(x, y, z)
        
        x = scene.x_.data
        y = scene.y_.data
        z = np.maximum(scene.surface_precip.data, 1e-2)
        axs[i, 2].pcolormesh(x, y, z, norm=precip_norm)
