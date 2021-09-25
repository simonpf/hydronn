"""
===================
hydronn.colocations
===================

This module implements the functions used to co-locate GPM and GOES
observations.
"""
import numpy as np
import pyresample
from pyresample.geometry import SwathDefinition
from pyresample import kd_tree
import xarray as xr

from hydronn.definitions import ROI
from hydronn.data.gpm import GPMCMBFile
from hydronn.data.goes import (GOES16File,
                               LOW_RES_CHANNELS,
                               MED_RES_CHANNELS,
                               HI_RES_CHANNELS)

def add_channels(dataset):
    """
    Add missing channels to dataset.

    Args:
        dataset: 'xarray.Dataset' containing GOES colocations.

    Return:
        The same dataset with missing channels filled with NAN.
    """
    for i in range(1, 16):
        v = f"C{i:02}"
        if v not in dataset.variables:
            if i in LOW_RES_CHANNELS:
                data = np.zeros((256, 256), dtype=np.float32)
                data[:] = np.nan
                dataset[v] = (("x", "y"), data)
            elif i in MED_RES_CHANNELS:
                data = np.zeros((512, 512), dtype=np.float32)
                data[:] = np.nan
                dataset[v] = (("x_1000", "y_1000"), data)
            elif i in HI_RES_CHANNELS:
                data = np.zeros((1024, 1024), dtype=np.float32)
                data[:] = np.nan
                dataset[v] = (("x_500", "y_500"), data)
    return dataset


def extract_colocations(gpm_file,
                        cache):
    """
    Extract GOES/GPM colocations for a give GPMCMB file.

    Args:
        gpm_file: GPMCMBFile for which to extract colocations.
        cache: The folder to use store downloaded data.

    Return:
        ``xarray.Dataset`` containing scenes of co-located GOES and
        GPMCMB files.
    """
    scenes = []
    for s in gpm_file.extract_scenes(ROI, 110):
        lat_0 = s.latitude.min().item()
        lat_1 = s.latitude.max().item()
        lat_c = 0.5 * (lat_0 + lat_1)
        lon_0 = s.longitude.min().item()
        lon_1 = s.longitude.max().item()
        lon_c = 0.5 * (lon_0 + lon_1)

        lon_0 = lon_c - 4
        lat_0 = lat_c - 4
        lon_1 = lon_c + 4
        lat_1 = lat_c + 4
        t = s.scan_time[64].item()

        goes_file = GOES16File.download(t, cache=cache)
        if goes_file is None:
            continue

        scene = goes_file.scene

        if all([c < 6 for c in goes_file.channels]):
            continue
        channel = [c for c in goes_file.channels if c >= 6][0]

        datasets = [f"C{c:02}" for c in range(1, 17)] + ["true_color"]
        scene.load(datasets)
        scene = scene.crop(ll_bbox=(lon_0, lat_0, lon_1, lat_1))

        area = scene[f"C{channel:02}"].attrs["area"]
        lons, lats = area.get_lonlats()
        gpm_swath = SwathDefinition(
            lats=s.latitude.data,
            lons=s.longitude.data
        )
        surface_precip = kd_tree.resample_nearest(
            gpm_swath,
            s.surface_precip.data,
            area,
            radius_of_influence=4.0e3,
            epsilon=0.01,
            fill_value=np.nan)

        low_res_arrays = {}
        for c in goes_file.channels:
            if c in LOW_RES_CHANNELS:
                channel_name = f"C{c:02}"
                array = scene[channel_name].load()
                array.close()
                array.attrs = {}
                low_res_arrays[channel_name] = array
            if "true_color" in scene.all_composite_names():
                low_res_arrays["true_color"] = scene["true_color"]
        data_lr = xr.Dataset(low_res_arrays)
        data_lr["surface_precip"] = (("y", "x"), surface_precip)
        x_size = data_lr.x.size
        d_x = x_size - 256
        y_size = data_lr.y.size
        d_y = y_size - 256
        data_lr = data_lr[{
            "x": slice(d_x // 2 + d_x % 2, x_size - d_x // 2),
            "y": slice(d_y // 2 + d_y % 2, y_size - d_y // 2)
        }]
        datasets = [data_lr.reset_index(["x", "y"])]

        med_res_arrays = {}
        for c in MED_RES_CHANNELS:
            if c in goes_file.channels:
                channel_name = f"C{c:02}"
                array = scene[channel_name].load()
                array.close()
                array.attrs = {}
                med_res_arrays[channel_name] = array
        if med_res_arrays:
            data_mr = xr.Dataset(med_res_arrays)
            x_size = data_mr.x.size
            d_x = x_size - 512
            y_size = data_mr.y.size
            d_y = y_size - 512
            data_mr = data_mr[{
                "x": slice(d_x // 2 + d_x % 2, x_size - d_x // 2),
                "y": slice(d_y // 2 + d_y % 2, y_size - d_y // 2)
            }].rename({"x": "x_1000", "y": "y_1000"})
            datasets.append(data_mr.reset_index(["x_1000", "y_1000"]))

        hi_res_arrays = {}
        for c in HI_RES_CHANNELS:
            if c in goes_file.channels:
                channel_name = f"C{c:02}"
                array = scene[channel_name].load()
                array.close()
                array.attrs = {}
                hi_res_arrays[channel_name] = array

        if hi_res_arrays:
            data_hr = xr.Dataset(hi_res_arrays)
            x_size = data_hr.x.size
            d_x = x_size - 1024
            y_size = data_hr.y.size
            d_y = y_size - 1024
            data_hr = data_hr[{
                "x": slice(d_x // 2 + d_x % 2, x_size - d_x // 2),
                "y": slice(d_y // 2 + d_y % 2, y_size - d_y // 2)
            }].rename({"x": "x_500", "y": "y_500"})
            datasets.append(data_hr.reset_index("x_500", "y_500"))

        dataset = add_channels(xr.merge(datasets))
        scenes.append(dataset)
    if not scenes:
        return None
    return xr.concat(scenes, dim="scenes", coords="all", join="override").drop("crs")
