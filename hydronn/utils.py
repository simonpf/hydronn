"""
=============
hydronn.utils
=============

Various utility functions.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr


def save_and_compress(data, filename):
    """
    Save 'xarray.Dataset' to file and compress.

    Note: The suffix '.gz' will be added to the filename by
    gzip so it should not be added to the filename.

    Args:
        data: The 'xarray.Dataset' to save.
        filename: The filename to store the file to.


    """
    data.to_netcdf(filename)
    subprocess.run(["gzip", filename], check=True)


def decompress_and_load(filename):
    """
    Load a potentially gzipped NetCDF file and return the
    data as 'xarray.Dataset'.

    Args:
        filename: The filename to store the file to.

    Return:
        An 'xarray.Dataset' containing the loaded data.
    """
    filename = Path(filename)
    if not filename.exists():
        if Path(filename).suffix == ".gz":
            raise ValueError(
                f"The file '{filename}' doesn't exist. "
            )
        else:
            filename_gz = Path(str(filename) + ".gz")
            if not filename_gz.exists():
                raise ValueError(
                    f"Neither the file '{filename}' nor '{filename}.gz' exist."
            )
            filename = filename_gz

    if Path(filename).suffix == ".gz":
        with TemporaryDirectory() as tmp:
            tmpfile = Path(tmp) / filename.stem
            with open(tmpfile, "wb") as decompressed:
                subprocess.run(
                    ["gunzip", "-c", str(filename)],
                    stdout=decompressed,
                    check=True
                )
            data = xr.load_dataset(tmpfile)
            Path(tmpfile).unlink()
    else:
        data = xr.open_dataset(filename)
    return data

def plot_scenes(dataset, n=3):

    n_scenes = dataset.scenes.size
    indices = np.random.permutation(n_scenes)

    f, axs = plt.subplots(n, 3, figsize=(12, n * 4))

    precip_norm = LogNorm(1e-1, 1e2)

    for i in range(n):
        scene = dataset[{"scenes": indices[i]}]

        x = scene.x_500_.data
        y = scene.y_500_.data
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
        try:
            axs[i, 1].contour(x, y, z, norm=precip_norm, levels=[0.1], cmap="Reds")
        except:
            pass


def plot_sample(x, y, n=3):

    n_scenes = y.shape[0]
    indices = np.random.permutation(n_scenes)

    f, axs = plt.subplots(n, 4, figsize=(16, n * 4))

    low_res, med_res, hi_res = x
    low_res = low_res.numpy()
    med_res = med_res.numpy()
    hi_res = hi_res.numpy()
    sp = y.numpy()

    x = np.arange(257)
    y = np.arange(257)
    x_c = 0.5 * (x[1:] + x[:-1])
    y_c = 0.5 * (y[1:] + y[:-1])
    x_med = np.linspace(0, 256, 513)
    y_med = np.linspace(0, 256, 513)
    x_hi = np.linspace(0, 256, 1025)
    y_hi = np.linspace(0, 256, 1025)

    print(sp.shape)

    for i in range(0, n):

        precip_norm = LogNorm(1e-1, 1e2)
        img_norm = Normalize(-1, 1)
        levels = np.logspace(-1, 1, 5)

        axs[i, 0].pcolormesh(x, y, low_res[i, -1], cmap="Greys_r")
        print(sp[i].max())
        try:
            axs[i, 0].contour(
                x_c, y_c, sp[i], norm=precip_norm, levels=levels, cmap="plasma",
                alpha=0.5
            )
        except:
            pass

        axs[i, 1].pcolormesh(x, y, low_res[i, -4], cmap="Greys_r")
        print(sp[i].max())
        try:
            axs[i, 0].contour(
                x_c, y_c, sp[i], norm=precip_norm, levels=levels, cmap="plasma",
                alpha=0.5
            )
        except:
            pass

        axs[i, 2].pcolormesh(x_med, y_med, med_res[i, 0], cmap="Greys_r")
        try:
            axs[i, 1].contour(
                x_c, y_c, sp[i], norm=precip_norm, levels=levels, cmap="plasma",
                contour=0.5

            )
        except:
            pass

        axs[i, 3].pcolormesh(x_hi, y_hi, hi_res[i, 0], cmap="Greys_r")
        try:
            axs[i, 2].contour(
                x_c, y_c, sp[i], norm=precip_norm, levels=levels, cmap="plasma",
                contour=0.5
            )
        except:
            pass
