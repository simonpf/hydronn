from datetime import datetime, timedelta
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pansat.products.satellite.goes import goes_16_l1b_radiances_all_full_disk
from pansat.download.providers.goes_aws import GOESAWSProvider
from satpy import Scene

t1 = datetime(2021, 9, 2, 20, 0, 0)
t2 = datetime(2021, 9, 2, 20, 30, 0)


roi = [-85, -40, -30, 10]
lon_0 = roi[0]
lat_0 = roi[1]
lon_1 = roi[2]
lat_1 = roi[3]


LOW_RES_CHANNELS = [4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
MED_RES_CHANNELS = [1, 3, 5]
HI_RES_CHANNELS = [2]


def find_goes_16_l1b_files(path, recursive=True):
    """
    Find all files containing GOES-16 L1b data.

    Args:
        path: Path to the folder in which to look for GOES-16 data.
        recursive: Whether to also search in sub-folders.
    """
    if recursive:
        return list(Path(path).glob("**/OR_ABI-L1b-RadF-*.nc"))
    return list(Path(path).glob("OR_ABI-L1b-RadF-*.nc"))


def get_start_times(filenames):
    """
    Extract the discrete start times in a list of GOES files.

    Args:
        filenames: A list of GOES-16 files.

    Return:
        List containing 'datetime' objects representing the start
        time of the GOES-16 files found in the given folder.
    """
    start_times = set()
    for f in filenames:
        start_time = Path(f).name.split("_")[-3]
        start_time = datetime.strptime(start_time[1:-1], "%Y%j%H%M%S")
        start_times.add(start_time)
    return list(start_times)


def find_start_times(path):
    """
    Find separate start times of GOES-16 L1b files in given
    folder.

    Args:
        path: Path to the folder containing the GOES ABI files.

    Return:
        List containing 'datetime' objects representing the start
        time of the GOES-16 files found in the given folder.
    """
    filenames = find_goes_16_l1b_files(path, recursive=True)
    return get_start_times(filenames)


def get_channels(filenames):
    """
    Extract available channels from a list of GOES-16 L1b files.

    Args:
        filenames: List containing the paths to the GOES-16 L1b filenames.

    Return:
        List containing the indices of the available channels.
    """
    channels = set()
    for file in filenames:
        channel = int(file.name.split("_")[1][-2:])
        channels.add(channel)
    return list(channels)


class GOES16File:
    """
    Class to read GOES-16 L1b files.

    One GOES16File object reads in data from all available channels files
    for a given point in time.
    """

    @staticmethod
    def download(time,
                 cache=None,
                 no_cache=False):
        """
        Cached download of GOES files.

        Args:
            time: The time for which to download available GOES files.
            cache: Folder to store downloaded files and to look for already
                existing files.
            no_cache: Disable cache.

        Return:
            GOES16File object
        """
        time = pd.Timestamp(time).to_pydatetime()
        start_time = time - timedelta(minutes=5)
        end_time = start_time + timedelta(minutes=10)

        provider = GOESAWSProvider(goes_16_l1b_radiances_all_full_disk)
        files = []

        if cache is None:
            dest = Path(p.default_destination)
        else:
            dest = Path(cache)

        dest.mkdir(parents=True, exist_ok=True)

        try:
            filenames = provider.get_files_in_range(
                start_time,
                end_time,
                start_inclusive=False
            )
        except Exception:
            return None
        if not filenames:
            return None
        for f in filenames:
            path = dest / f
            if not path.exists() or no_cache:
                provider.download_file(f, path)
            files.append(path)
        return GOES16File(files)

    @staticmethod
    def open_files(files):
        """
        Combine files into GOES16File corresponding to discrete start
        times.


        Args:
            files: A list of GOES 16 L1b files.

        Return:
            One 'GOES16File' object for each separate start time found in the
            list of files.
        """
        start_times = get_start_times(files)
        gpm_files = []
        for time in start_times:
            time_s = time.strftime("%Y%j%H%M%S")
            pattern = re.compile(f"OR_ABI-L1b-RadF-[\w_]*s{time_s}[\w_]*.nc")
            channel_files = [f for f in files if pattern.match(f.name)]
            gpm_files.append(GOES16File(channel_files))
        return gpm_files

    @staticmethod
    def open_path(path):
        """
        Open all available GOES-16 L1b at a given path.

        Args:
            path: Folder in which to look for GOES-16 files. Search is
                performed recursively.

        Return:
            List of 'GOES16File' objects for the time points that were found
            in the provided directory.
        """
        start_times = find_start_times(path)
        files = []
        for time in start_times:
            time_s = time.strftime("%Y%j%H%M%S")
            pattern = f"OR_ABI-L1b-RadF-*s{time_s}*.nc"
            channel_files = list(
                Path(path).glob(f"**/{pattern}")
            )
            files.append(GOES16File(channel_files))
        return files

    def __init__(self, channel_files):
        start_times = get_start_times(channel_files)
        self.start_time = min(start_times)
        tds = [(t - self.start_time).total_seconds() for t in start_times]
        if max(tds) > 300:
            raise ValueError("Provided channel files have inconsistent start times.")
        self.channels = get_channels(channel_files)
        self.scene = Scene(map(str, channel_files), reader="abi_l1b")


    def __repr__(self):
        return f"GOES16File(channels={self.channels})"
