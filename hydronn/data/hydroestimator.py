"""
===========================
hydronn.data.hydroestimator
===========================

This module provides functions to load the hydroestimator data.
"""
from pathlib import Path

import numpy as np
import xarray as xr
import scipy
from scipy.interpolate import interp1d


class Correction:
    """
    A correction to apply to the raw hydroestimator data.

    The correction class loads the lookup table that defines the
    correction on creation. It then acts as a functor that can
    be applied to any array of rain rates returning the corrected
    rain rates.
    """
    def __init__(self, filename):
        """
        Load correction data.

        Args:
            filename:


        """
        self.lut = np.loadtxt(filename, skiprows=1)
        self.interp = interp1d(self.lut[:, 0], self.lut[:, 1])

    def __call__(self, x):
        """
        Apply correction to a array of rain rates.

        Args:
            x: The array of rainrates to which to apply the correction.

        Return:
            An array with the same shape as 'x' containing the corrected
            rain rates.
        """
        return self.interp(x)


def load_file(filename, correction=None):
    """
    Load hydroestimator data from a single file.

    Args:
        filename: The path of the file to load.

    Return:
        A 1613 x 1349 array containing the rainrates from the file.
    """
    M = 1613
    N = 1349
    precip = np.fromfile(filename, dtype="u2").reshape((M, N)) / 10.0
    precip = precip[::-1, :]

    if correction is not None:
        precip = correction(precip)

    return precip


def get_lats_and_lons():
    """
    Get array containing the latitudes and longitudes corresponding to the
    hydroestimator data.

    Return:
        A tuple ``(lats, lons)`` containing the latitudes and longitudes
        corresponding to the hydroestimator grid.
    """
    M = 1613
    # Calculate latitudes.
    lat_0 = -44.95
    d_lat = 0.0359477
    lats = lat_0 + np.arange(M) * d_lat
    # Calculate longitudes.
    lon_0 = 278.0 - 360
    d_lon = 0.0382513
    lons = lon_0 + np.arange(N) * d_lon
    return lats, lons


def get_date(filename):
    """
    Parse data from hydroestimator file name.

    Args:
       filename: The filename of the hydroestimator file.

    Return:
       ``np.datetime64`` object representing the timestamp of the
       given filename.
    """
    name = Path(filename).name
    time = name.split("_")[1]
    year = time[:4]
    month = time[4:6]
    day = time[6:8]
    hour = time[8:10]
    minute = time[10:12]
    s = f"{year}-{month}-{day}T{hour}:{minute}:00"
    return np.datetime64(s)


def load_data(data_path):
    """
    Load all hydroestimator data from the given path and return it as a 'xarray.Dataset'.

    Args:
        data_path: Folder containing the hydroestimator files to load.

    Return:
        A 'xarray.Dataset' containing the concatenated data from all files in
        the given folder.
    """
    files = sorted(list(Path(data_path).glob("*.bin")))
    data = []
    times = []
    for f in files:
        times.append(get_date(f))
        data.append(load_file(f))
    times = np.stack(times)
    data = np.stack(data)

    latitude, longitude = get_lats_and_lons()

    dataset = xr.Dataset({
        "latitude": latitude,
        "longitude": longitude,
        "time": time,
        "surface_precip": (("time", "latitude", "longitude"), data)
    })
    return dataset
