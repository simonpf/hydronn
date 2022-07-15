"""
===========================
hydronn.data.hydroestimator
===========================

This module provides functions to load the hydroestimator data.
"""
from pathlib import Path
import subprocess
import io
import re

import numpy as np
import xarray as xr
from pyresample import create_area_def
from pyresample.kd_tree import resample_nearest
from pyresample.geometry import SwathDefinition
import scipy
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

from hydronn.definitions import BRAZIL


HYDRO_GRID = create_area_def(
    'HYDRO',
    {'proj': 'longlat', 'datum': 'WGS84'},
    area_extent=[-82, -44.95, -30.3989963, 13.03364],
    resolution=(0.0382513, 0.0359477),
    units='degrees',
)


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
    filename = Path(filename)
    M = 1613
    N = 1349
    try:
        if filename.suffix == ".gz":
            decompressed = io.BytesIO()
            args = ["gunzip", "-c", str(filename)]
            with subprocess.Popen(args, stdout=subprocess.PIPE) as proc:
                decompressed.write(proc.stdout.read())
                precip = np.frombuffer(decompressed, dtype="u2").reshape((M, N))
        else:
            precip = np.fromfile(filename, dtype="u2").reshape((M, N))
        precip = precip.astype(np.float32) / 10.0
        precip = precip[::-1, :]
        if correction is not None:
            precip = correction(precip)
    except ValueError:
        precip = np.nan * np.ones((M, N), dtype=np.float32)

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
    N = 1349
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


class Hydroestimator:
    """
    Product class for representing hydrometeor files.
    """
    def __init__(self, correction=None):
        """
        Create product instance.

        Args:
            correction: Optional path of a CDF correction to apply to
                the data.
        """
        if correction is not None:
            correction = Correction(correction)
        self.correction = correction
        self.filename_regexp = re.compile("S\d{8}_\d{12}\.bin(\.gz){0,1}")

    def filename_to_date(self, filename):
        """
        Extract date from filename.

        Args:
            filename: Filename of a file containing Hydrometeor results.

        Return:
            ``datetime.datetime`` object representing the start time of
            the given Hydrometeor file.
        """
        return get_date(filename)

    def open(self, filename):
        """
        Load the Hydrometeor results into an ``xarray.Dataset``.

        Args:
            filename: Name of a file containing Hydrometeor results.

        Return:
            The Hydrometeor results as ``xarray.Dataset``.
        """
        precip = load_file(filename, correction=self.correction)
        lats, lons = get_lats_and_lons()

        precip_r = resample_nearest(
            HYDRO_GRID,
            precip[::-1],
            BRAZIL,
            radius_of_influence=5e3,
            fill_value=np.nan,
        )

        lons, lats = BRAZIL.get_lonlats()
        lons = lons[0]
        lats = lats[::-1, 0]

        dataset = xr.Dataset({
            "latitude": (("latitude",), lats),
            "longitude": (("longitude",), lons),
            "surface_precip": (("latitude", "longitude"), precip_r)
        })
        return dataset


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

    dataset = xr.Dataset(
        {
            "latitude": latitude,
            "longitude": longitude,
            "time": time,
            "surface_precip": (("time", "latitude", "longitude"), data),
        }
    )
    return dataset


def load_and_interpolate_data(data_path, gauge_data, correction=None):
    """
    Calculated accumulated and mean precipitation for the hydroestimator data.

    Args:
        data_path: Folder containing the hydroestimator files to load.

    Return:
        A 'xarray.Dataset' containing  accumulated and mean precipitation.
    """
    if correction is not None:
        correction = Correction(correction)

    files = sorted(list(Path(data_path).glob("*.bin")))

    latitude, longitude = get_lats_and_lons()
    m = latitude.size
    n = longitude.size
    values = np.zeros((m, n), dtype=np.float32)

    interpolator = RegularGridInterpolator(
        (latitude, longitude), values, bounds_error=False
    )
    x = np.stack((gauge_data.latitude.data, gauge_data.longitude.data), axis=-1)

    precip = []
    times = []

    for f in files:
        times.append(get_date(f))
        data = load_file(f, correction=correction)
        interpolator.values = data
        precip.append(interpolator(x))
        print(f)

    times = np.stack(times)
    precip = np.stack(precip, axis=-1)

    dataset = xr.Dataset(
        {
            "gauges": (("gauges",), gauge_data.gauges.data),
            "time": (("time",), times),
            "surface_precip": (("gauges", "time"), precip),
        }
    )
    return dataset


def calculate_accumulations(data_path, correction=None, start=None, end=None):
    """
    Calculated accumulated and mean precipitation for the hydroestimator data.

    Args:
        data_path: Folder containing the hydroestimator files to load.
        correction: Path to a text file specifying a quantile matching
            correction for the Hydroestimator data.
        start: Optional numpy.datetime64 specifying start of a time
            interval over which to accumulate the precipitation.
        end: Optional numpy.datetime64 specifying end of a time
            interval over which to accumulate the precipitation.

    Return:
        A 'xarray.Dataset' containing  accumulated and mean precipitation.
    """
    if correction is not None:
        correction = Correction(correction)

    files = sorted(list(Path(data_path).glob("*.bin")))

    latitude, longitude = get_lats_and_lons()
    m = latitude.size
    n = longitude.size
    acc = np.zeros((m, n), dtype=np.float32)
    counts = np.zeros((m, n), dtype=np.float32)

    for f in files:

        date = f.name.split("_")[1].split(".")[0]
        year = date[:4]
        month = date[4:6]
        day = date[6:8]
        hour = date[8:10]
        minute = date[10:12]
        date = np.datetime64(f"{year}-{month}-{day}T{hour}:{minute}:00")
        print(date)
        if date < start or date >= end:
            continue

        data = load_file(f, correction=correction)
        data = np.nan_to_num(data, 0.0)
        acc += data
        counts += (data >= 0).astype(np.float32)

    mean = acc / counts
    dataset = xr.Dataset(
        {
            "latitude": latitude,
            "longitude": longitude,
            "surface_precip_acc": (("latitude", "longitude"), acc),
            "surface_precip_mean": (("latitude", "longitude"), mean),
            "counts": (("latitude", "longitude"), counts),
        }
    )
    return dataset
