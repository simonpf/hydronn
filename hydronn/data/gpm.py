"""
================
hydronn.data.gpm
================

This module provides functions to read level 2B files of the GPM
combined product, which is used to train they hydronn algorithm.
"""
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from h5py import File
import numpy as np
import pandas as pd
import xarray as xr
from pansat.download.providers.ges_disc import GesdiscProvider


GPM_FILES = open(Path(__file__).parent / "gpm_files.txt").readlines()
GPM_FILES = [name.strip() for name in GPM_FILES]

def get_gpm_files(year, day):
    """
    Return list of GPM files for given year and day of month.

    Args:
        year: The year.
        day: The day of the month, i.e. in [1, 31].

    Return:
        List containing all GPM CMB files over Brazil in the time range
        between December 2017 and September 2021.
    """
    return GPM_FILES_SORTED.get((year, day), [])


def extract_scenes(dataset, scans_per_scene):
    """
    Split dataset along scene dimension.

    Args:
        dataset: A dataset with a "scans" dimension.
        scans_per_scene: How many scans to include in each scene.

    Return:
        Iterator over the scenes in the file.
    """
    tot_scans = dataset.scans.size
    n_scenes = tot_scans / scans_per_scene
    residual = tot_scans % scans_per_scene
    offset = np.random.randint(0, residual)
    i_start = offset
    i_end = i_start + scans_per_scene
    while i_end < tot_scans:
        yield dataset[{"scans": slice(i_start, i_end)}]
        i_start += scans_per_scene
        i_end = i_start + scans_per_scene


class GPMCMBFile:
    """
    Class to read in GPM combined data.
    """

    @staticmethod
    def download(url, destination=None):

        if destination is None:
            destination = Path(".")
        else:
            destination = Path(destination)

        path = urlparse(url).path
        destination = destination / Path(path).name
        GesdiscProvider.download_url(url, destination)
        return GPMCMBFile(destination)

    def __init__(self, filename):
        """
        Create GPMCMB object to read a given file.

        Args:
            filename: Path pointing to the file to read.

        """
        self.filename = Path(filename)
        time = self.filename.stem.split(".")[4][:-8]
        self.start_time = datetime.strptime(
            time,
            "%Y%m%d-S%H%M%S"
        )

    def to_xarray_dataset(self, roi=None):
        """
        Load data in file into 'xarray.Dataset'.

        Args:
            roi: Optional bounding box given as list
                 ``[lon_0, lat_0, lon_1, lat_1]`` specifying the longitude
                 and latitude coordinates of the lower left
                 (``lon_0, lat_0``) and upper right (``lon_1, lat_1``)
                 corners. If given, only scans containing at least one pixel
                 within the given bounding box will be returned.
        """
        with File(str(self.filename), "r") as data:

            data = data['MS']
            latitude = data["Latitude"][:]
            longitude = data["Longitude"][:]

            if roi is not None:
                lon_0, lat_0, lon_1, lat_1 = roi
                inside = ((longitude >= lon_0) *
                          (latitude >= lat_0) *
                          (longitude < lon_1) *
                          (latitude < lat_1))
                inside = np.any(inside, axis=1)
                i_start, i_end = np.where(inside)[0][[0, -1]]
            else:
                i_start = 0
                i_end = latitude.shape[0]

            latitude = latitude[i_start:i_end]
            longitude = longitude[i_start:i_end]

            date = {
                "year": data["ScanTime"]["Year"][i_start:i_end],
                "month": data["ScanTime"]["Month"][i_start:i_end],
                "day": data["ScanTime"]["DayOfMonth"][i_start:i_end],
                "hour": data["ScanTime"]["Hour"][i_start:i_end],
                "minute": data["ScanTime"]["Minute"][i_start:i_end],
                "second": data["ScanTime"]["Second"][i_start:i_end]
            }
            date = pd.to_datetime(date)
            surface_precip = data["surfPrecipTotRate"][i_start:i_end]

            dataset = xr.Dataset({
                "scan_time": (("scans",), date),
                "latitude": (("scans", "pixels"), latitude),
                "longitude": (("scans", "pixels"), longitude),
                "surface_precip": (("scans", "pixels"), surface_precip)
            })
            return dataset

    def extract_scenes(self, roi, scans_per_scene):
        """
        Extract scenes in given region of interest (ROI).

        Args:
            roi: Region of interest ``[lon_0, lat_0, lon_1, lat_1]`` defined
                by the longitude and latitude coordinates of the lower-left
                and upper-right corner of a bounding box.
            scans_per_scene: How many scans to include in each scene.

        Return:
            Iterator over the scenes in the file.
        """
        dataset = self.to_xarray_dataset(roi=roi)
        return extract_scenes(dataset, scans_per_scene)

GPM_FILES_SORTED = {}
for f in GPM_FILES:
    gpm_file = GPMCMBFile(f)
    year = gpm_file.start_time.year
    day = gpm_file.start_time.day
    GPM_FILES_SORTED.setdefault((year, day), []).append(f)
