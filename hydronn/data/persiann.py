"""
hydronn.data.persiann
=====================

This module provides a simple interface to load precipitation data
from PERSIANN product files.
"""
from pathlib import Path

import numpy as np
import xarray as xr
from pansat.products.satellite.persiann import CCS

from hydronn import definitions


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
        print(filename)
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
            print(date)
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
