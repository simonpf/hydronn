"""
=============
hydronn.utils
=============

Various utility functions.
"""
import io
import gzip

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
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
    subprocess.run(["gzip", "-f", filename], check=True)


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
            raise ValueError(f"The file '{filename}' doesn't exist. ")
        elif Path(filename).suffix == ".lz4":
            raise ValueError(f"The file '{filename}' doesn't exist. ")
        else:
            filename_gz = Path(str(filename) + ".gz")
            if not filename_gz.exists():
                filename_lz4 = Path(str(filename) + ".lz4")
                if not filename_lz4.exists():
                    raise ValueError(
                        f"Neither the file '{filename}' nor '{filename}.gz' exist."
                    )
                filename = filename_lz4
            else:
                filename = filename_gz

    if Path(filename).suffix == ".gz":
        decompressed = io.BytesIO()
        args = ["gunzip", "-c", str(filename)]
        with subprocess.Popen(args, stdout=subprocess.PIPE) as proc:
            decompressed.write(proc.stdout.read())
        decompressed.seek(0)
        data = xr.load_dataset(decompressed, engine="h5netcdf")
    elif Path(filename).suffix == ".lz4":
        decompressed = io.BytesIO()
        args = ["unlz4", str(filename)]
        with subprocess.Popen(args, stdout=subprocess.PIPE) as proc:
            decompressed.write(proc.stdout.read())
        decompressed.seek(0)
        data = xr.load_dataset(decompressed, engine="h5netcdf")
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
                x_c,
                y_c,
                sp[i],
                norm=precip_norm,
                levels=levels,
                cmap="plasma",
                alpha=0.5,
            )
        except:
            pass

        axs[i, 1].pcolormesh(x, y, low_res[i, -4], cmap="Greys_r")
        print(sp[i].max())
        try:
            axs[i, 0].contour(
                x_c,
                y_c,
                sp[i],
                norm=precip_norm,
                levels=levels,
                cmap="plasma",
                alpha=0.5,
            )
        except:
            pass

        axs[i, 2].pcolormesh(x_med, y_med, med_res[i, 0], cmap="Greys_r")
        try:
            axs[i, 1].contour(
                x_c,
                y_c,
                sp[i],
                norm=precip_norm,
                levels=levels,
                cmap="plasma",
                contour=0.5,
            )
        except:
            pass

        axs[i, 3].pcolormesh(x_hi, y_hi, hi_res[i, 0], cmap="Greys_r")
        try:
            axs[i, 2].contour(
                x_c,
                y_c,
                sp[i],
                norm=precip_norm,
                levels=levels,
                cmap="plasma",
                contour=0.5,
            )
        except:
            pass


def adapt_gauge_precip(precip):
    result = precip.copy()
    zeros = precip < 0.2
    result[zeros] = np.random.uniform(1e-3, 5e-3, size=zeros.sum())
    non_zeros = precip >= 0.2
    result[non_zeros] += np.random.uniform(-0.1, 0.1, size=non_zeros.sum())
    return result


def load_style():
    """
    Load hydronn plotting style.
    """
    path = Path(__file__).parent / "files" / "hydronn.mplstyle"
    plt.style.use(path)


def scale_bar(
        ax,
        length,
        location=(0.5, 0.05),
        linewidth=3,
        height=0.01,
        border=0.05,
        parts=4
):
    """
    Draw a scale bar on a cartopy map.

    Args:
        ax: The matplotlib.Axes object to draw the axes on.
        length: The length of the scale bar in meters.
        location: A tuple ``(h, w)`` defining the fractional horizontal
            position ``h`` and vertical position ``h`` in the given axes
            object.
        linewidth: The width of the line.
    """
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(ccrs.PlateCarree())

    lon_c = lon_min + (lon_max - lon_min) * location[0]
    lat_c = lat_min + (lat_max - lat_min) * location[1]
    transverse_merc = ccrs.TransverseMercator(lon_c, lat_c)

    x_min, x_max, y_min, y_max = ax.get_extent(transverse_merc)

    x_c = x_min + (x_max - x_min) * location[0]
    y_c = y_min + (y_max - y_min) * location[1]

    x_left = x_c - length / 2
    x_right = x_c  + length / 2

    def to_axes_coords(point):
        crs = ax.projection
        p_data = crs.transform_point(*point, src_crs=transverse_merc)
        return ax.transAxes.inverted().transform(ax.transData.transform(p_data))

    def axes_to_lonlat(point):
        p_src = ax.transData.inverted().transform(ax.transAxes.transform(point))
        return ccrs.PlateCarree().transform_point(*p_src, src_crs=ax.projection)


    left_ax = to_axes_coords([x_left, y_c])
    right_ax = to_axes_coords([x_right, y_c])

    print("LEFT:  ", axes_to_lonlat(left_ax))
    print("RIGHT: ", axes_to_lonlat(right_ax))

    l_ax = right_ax[0] - left_ax[0]
    l_part = l_ax / parts



    left_bg = [
        left_ax[0] - border,
        left_ax[1] - height / 2 - border
    ]

    background = Rectangle(
        left_bg,
        l_ax + 2 * border,
        height + 2 * border,
        facecolor="none",
        transform=ax.transAxes,
    )
    ax.add_patch(background)

    for i in range(parts):
        left = left_ax[0] + i * l_part
        bottom = left_ax[1] - height / 2

        color = "k" if i % 2 == 0 else "w"
        rect = Rectangle(
            (left, bottom),
            l_part,
            height,
            facecolor=color,
            edgecolor="k",
            transform=ax.transAxes,
        )
        ax.add_patch(rect)





    x_bar = [x_c - length / 2, x_c + length / 2]
    x_text = 0.5 * (left_ax[0] + right_ax[0])
    y_text = left_ax[1] + 0.5 * height + 2 * border
    ax.text(x_text,
            y_text,
            f"{length / 1e3:g} km",
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='bottom'
    )
