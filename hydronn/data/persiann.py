"""
hydronn.data.persiann
=====================

This module provides a simple interface to load precipitation data
from PERSIANN product files.
"""
from pathlib import Path

import numpy as np
import xarray as xr
from pansat.time import to_datetime64
from pyresample import create_area_def
from pyresample.kd_tree import resample_nearest
from pansat.products.satellite.persiann import CCS
from hydronn import definitions
from hydronn.definitions import BRAZIL

CCS_GRID = create_area_def(
    'IMERG',
    {'proj': 'longlat', 'datum': 'WGS84'},
    area_extent=[-180, -60, 180, 60],
    resolution=0.04,
    units='degrees',
)

def resample(filename):
    """
    Resample PERSIANN data to GOES 4 km grid over Brazil.

    Return:
        An ``xarray.Dataset`` containing the resampled observations.
    """
    product = CCS()
    data = product.open(filename).rename({"precipitation": "surface_precip"})
    left = data[{"longitude": slice(4500, None)}]
    right = data[{"longitude": slice(0, 4500)}]
    data = xr.concat([left, right], dim="longitude")
    longitude_new = data.longitude.data
    longitude_new[:4500] -= 360
    data["longitude"] = longitude_new

    precip = data["surface_precip"].data

    lons, lats = CCS_GRID.get_lonlats()

    precip_r = resample_nearest(
        CCS_GRID,
        precip[0],
        BRAZIL,
        radius_of_influence=5e3,
        fill_value=np.nan,
    )

    lons, lats = BRAZIL.get_lonlats()
    lons = lons[0]
    lats = lats[:, 0]

    time = to_datetime64(product.filename_to_date(filename))

    dataset = xr.Dataset({
        "latitude": (("latitude",), lats),
        "longitude": (("longitude",), lons),
        "time": (("time",), [time]),
        "surface_precip": (("time", "longitude", "latitude"),
                           precip_r[np.newaxis]),
    })

    return dataset


def load_and_interpolate_data(path, gauge_data):
    """
    Load and interpolate PERSIANN data to gauge coordinates.

    Args:
        path: Path to the folder containing the PERSIANN data.
        gauge_data: xarray.Dataset containing the gauge data.

    Return:
       xarray.Dataset containing the loaded PERSIANN data interpolated
       to the coordinates of the gauges.
    """
    files = sorted(list(Path(path).glob("*.bin.gz")))
    product = CCS()
    datasets = []
    for filename in files:
        data = product.open(filename).rename({"precipitation": "surface_precip"})
        datasets.append(
            data.interp(
                {
                    "latitude": gauge_data.latitude,
                    "longitude": gauge_data.longitude + 360,
                }
            )
        )
    dataset = xr.concat(datasets, dim="time")
    return dataset


def calculate_accumulations(path, start=None, end=None):
    """
    Calculate precip mean and accumulated precip from PERSIANN data.

    Args:
        path: Path to the folder containing the PERSIANN data.
        start: Optional numpy.datetime64 specifying start of a time
            interval over which to accumulate the precipitation.
        end: Optional numpy.datetime64 specifying end of a time
            interval over which to accumulate the precipitation.

    Return:
       xarray.Dataset containing the accumulated and average precip
       rates over brazil.
    """
    files = sorted(list(Path(path).glob("*.bin.gz")))
    product = CCS()
    dataset = None
    for filename in files:

        if start is not None and end is not None:
            date = filename.name.split(".")[0]
            year = date[-7:-5]
            doy = date[-5:-2]
            hour = date[-2:]

            date = np.datetime64(f"20{year}-01-01T00:00:00") + np.array(
                int(doy) * 24 + int(hour)
            ).astype("timedelta64[h]")
            if date < start or date >= end:
                continue

        data = product.open(filename)
        lon_0, lat_0, lon_1, lat_1 = definitions.ROI
        lon_0 = lon_0 + 360
        lon_1 = lon_1 + 360
        lon_indices = (data.longitude.data >= lon_0) * (data.longitude.data < lon_1)
        lat_indices = (data.latitude.data >= lat_0) * (data.latitude.data < lat_1)
        data = data[{"latitude": lat_indices, "longitude": lon_indices}]
        precip = np.nan_to_num(data.precipitation.data[0], 0.0)
        counts = np.isfinite(data.precipitation.data[0]).astype(np.float32)
        if dataset is None:
            dataset = xr.Dataset(
                {
                    "latitude": (("latitude",), data.latitude.data),
                    "longitude": (("longitude"), data.longitude.data),
                    "surface_precip_acc": (("latitude", "longitude"), precip),
                    "counts": (("latitude", "longitude"), counts),
                }
            )
        else:
            dataset.surface_precip_acc.data += precip
            dataset.counts.data += counts
    dataset["surface_precip_mean"] = (
        ("latitude", "longitude"),
        dataset.surface_precip_acc.data / dataset.counts.data,
    )
    return dataset
