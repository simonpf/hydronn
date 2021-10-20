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

class ImergFile:

    @classmethod
    def load_files(cls,
                   path,
                   start_time=None,
                   end_time=None,
                   roi=None):
        path = Path(path)
        files = list(path.glob("3B-HHR.MS.MRG*.HDF5"))

        start_times = []
        for f in files:
            parts = f.stem.split(".")[4].split("-")[:2]
            start_times.append(datetime.strptime(
                "".join(parts),
                "%Y%m%dS%H%M%S"
            ))

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

        dataset = xr.Dataset({
            "latitude": (("latitude",), lats),
            "longitude": (("longitude",), lons),
            "time": (("time",), [time]),
            "lat_bounds": (("lat_bounds"), lat_bounds),
            "lon_bounds": (("lon_bounds"), lon_bounds),
            "surface_precip": (("time", "longitude", "latitude"), precip)
        })

        return dataset
