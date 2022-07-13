"""
==================
hydronn.data.imerg
==================

Interface to load IMERG data into xarray.datasets.
"""
from datetime import datetime, timedelta
from pathlib import Path

from h5py import File
import numpy as np
import pandas as pd
import xarray as xr
from pyresample import create_area_def
from pyresample.kd_tree import resample_nearest

from hydronn.definitions import ROI, BRAZIL


###############################################################################
# ImergFile
###############################################################################


IMERG_GRID = create_area_def(
    'IMERG',
    {'proj': 'longlat', 'datum': 'WGS84'},
    area_extent=[-180, -90, 180, 90],
    resolution=0.1,
    units='degrees',
)

class ImergFile:
    """
    Interface class to read IMERG data.
    """

    @classmethod
    def load_files(cls, path, start_time=None, end_time=None, roi=None):
        """
        Load all files from a given folder and concatenate results.

        Args:
            path: Path pointing to a folder containing the IMERG files.
            start_time: If given, can be used to specify the beginning
                of a time range for which to the files.
            end_time: If given, can be used to specify the end of a time range
                for which to read the files.
            roi: A region of interest (ROI) given by the coordinates of its
                corners ``(lon_ll, lat_ll, lon_ur, lat_ur)``.

        Return:
            A 'xarray.Dataset' containing the data from the loaded files
            concatenated along the time dimension.
        """
        path = Path(path)
        files = list(path.glob("3B-HHR.MS.MRG*.HDF5"))

        start_times = []
        for f in files:
            parts = f.stem.split(".")[4].split("-")[:2]
            start_times.append(datetime.strptime("".join(parts), "%Y%m%dS%H%M%S"))

        matching = []
        if start_time is not None and end_time is not None:
            for t, f in zip(start_times, files):
                if t >= start_time and t < end_time:
                    matching.append(f)
                    matching_times.append(t)
        else:
            matching = files

        imerg_files = [cls(f).to_xarray_dataset(roi=roi) for f in matching]
        imerg_files = sorted(imerg_files, key=lambda x: x.time[0])
        return xr.concat(imerg_files, "time")

    def __init__(self, filename):
        self.filename = Path(filename)

        date = self.filename.stem.split(".")[4]
        parts = date.split("-")
        date = datetime.strptime(parts[0], "%Y%m%d")
        start = datetime.strptime(parts[0] + parts[1], "%Y%m%dS%H%M%S")
        end = datetime.strptime(parts[0] + parts[2], "%Y%m%dE%H%M%S")
        self.start_time = start
        self.end_time = end

    def to_xarray_dataset(self, roi=None):
        """
        Load data into 'xarray.Dataset'.

        Args:
           roi: If given, should contain the coordinates
               ``(lon_ll, lat_ll, lon_ur, lat_ur)`` of a
               rectangular bounding box to which the loaded
               will be restricted.

        Return:
            A 'xarray.Dataset' containing the loaded data from the
            file.
        """
        data = File(str(self.filename), "r")
        lats = data["Grid/lat"][:]
        lons = data["Grid/lon"][:]

        if roi is not None:
            lon_0, lat_0, lon_1, lat_1 = roi
            i_start = np.where(lons > lon_0)[0][0]
            i_end = np.where(lons > lon_1)[0][0]
            j_start = np.where(lats > lat_0)[0][0]
            j_end = np.where(lats > lat_1)[0][0]
        else:
            i_start = 0
            i_end = 3600
            j_start = 0
            j_end = 1800

        lons = lons[i_start:i_end]
        lats = lats[j_start:j_end]
        lat_bounds = data["Grid/lat_bnds"][j_start:j_end]
        lat_bounds = np.concatenate([lat_bounds[:, 0], lat_bounds[-1:, 1]])
        lon_bounds = data["Grid/lon_bnds"][i_start:i_end]
        lon_bounds = np.concatenate([lon_bounds[:, 0], lon_bounds[-1:, 1]])
        precip = data["Grid/precipitationCal"][:, i_start:i_end, j_start:j_end]

        start = pd.Timestamp(self.start_time).to_datetime64()
        end = pd.Timestamp(self.end_time).to_datetime64()
        time = start + 0.5 * (end - start)

        dataset = xr.Dataset(
            {
                "latitude": (("latitude",), lats),
                "longitude": (("longitude",), lons),
                "time": (("time",), [time]),
                "lat_bounds": (("lat_bounds"), lat_bounds),
                "lon_bounds": (("lon_bounds"), lon_bounds),
                "surface_precip": (("time", "longitude", "latitude"), precip),
            }
        )

        return dataset

    def resample(self):
        """
        Resample IMERG data to GOES 4 km grid over Brazil.

        Return:
            An ``xarray.Dataset`` containing the resampled observations.
        """
        with File(str(self.filename), "r") as data:
            precip = data["Grid/precipitationCal"][:]
            precip_r = resample_nearest(
                IMERG_GRID,
                precip[0].T[::-1],
                BRAZIL,
                radius_of_influence=10e3,
                fill_value=np.nan,
            )

        lons, lats = BRAZIL.get_lonlats()
        lons = lons[0]
        lats = lats[::-1, 0]

        start = pd.Timestamp(self.start_time).to_datetime64()
        end = pd.Timestamp(self.end_time).to_datetime64()
        time = start + 0.5 * (end - start)

        dataset = xr.Dataset(
            {
                "latitude": (("latitude",), lats),
                "longitude": (("longitude",), lons),
                "time": (("time",), [time]),
                "surface_precip": (("time", "longitude", "latitude"),
                                   precip_r[np.newaxis]),
            }
        )
        return dataset


def load_and_interpolate_data(path, gauge_data):
    """
    Load and interpolate IMERG data to gauge coordinates.

    Args:
        path: Path to the folder containing the IMERG data.
        gauge_data: xarray.Dataset containing the gauge data.

    Return:
       xarray.Dataset containing the loaded IMERG data interpolated
       to the coordinates of the gauges.
    """
    files = sorted(list(Path(path).glob("*.HDF5")))
    datasets = []
    for filename in files:
        data = ImergFile(filename).to_xarray_dataset(roi=ROI)
        datasets.append(
            data.interp(
                {
                    "latitude": gauge_data.latitude,
                    "longitude": gauge_data.longitude,
                }
            )
        )
    dataset = xr.concat(datasets, dim="time")
    return dataset


def calculate_accumulations(path, start=None, end=None):
    """
    Calculate accumulated precipitation.

    Args:
        path: Path to the folder containing the IMERG data.
        start: Optional numpy.datetime64 specifying start of a time
            interval over which to accumulate the precipitation.
        end: Optional numpy.datetime64 specifying end of a time
            interval over which to accumulate the precipitation.

    Return:
       xarray.Dataset containing the IMERG data accumulated
       over the range of available files.
    """
    files = sorted(list(Path(path).glob("*.HDF5")))
    results = None
    for filename in files:

        if start is not None and end is not None:
            date = filename.name.split(".")[4]
            year = date[:4]
            month = date[4:6]
            day = date[6:8]
            hour = date[10:12]
            minute = date[12:14]
            second = date[14:16]
            date = np.datetime64(f"{year}-{month}-{day}T{hour}:{minute}:{second}")
            if date < start or date >= end:
                continue

        data = ImergFile(filename).to_xarray_dataset(roi=ROI)
        if results is None:
            precip = data.surface_precip.data[0]
            counts = 0.5 * (precip >= 0).astype(np.float32)
            lons = data.longitude.data
            lats = data.latitude.data

            results = xr.Dataset(
                {
                    "latitude": (("latitude",), lats),
                    "longitude": (("longitude",), lons),
                    "counts": (("longitude", "latitude"), counts),
                    "surface_precip": (("longitude", "latitude"), precip),
                }
            )
        else:
            precip = data.surface_precip.data[0]
            counts = 0.5 * (precip >= 0).astype(np.float32)
            results.surface_precip.data += 0.5 * precip
            results.counts.data += 0.5 * (precip >= 0).astype(np.float32)

    results = results.rename({"surface_precip": "surface_precip_acc"})
    mean = results.surface_precip_acc.data / results.counts.data
    results["surface_precip_mean"] = (("longitude", "latitude"), mean)

    return results
