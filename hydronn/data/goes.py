from datetime import datetime, timedelta
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pansat.products.satellite.goes import goes_16_l1b_radiances_all_full_disk
from pansat.download.providers.goes_aws import GOESAWSProvider
from satpy import Scene
import torch
import xarray as xr


LOW_RES_CHANNELS = [4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
MED_RES_CHANNELS = [1, 3, 5]
HI_RES_CHANNELS = [2]


ROW_START = 2450
ROW_END = ROW_START + 1920
COL_START = 2750
COL_END = COL_START + 1920


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
        """
        Load GOES-16 observations.

        Args:
            channel_files: The files containing the GOES observations
                for each of the channels.
        """
        start_times = get_start_times(channel_files)
        self.start_time = min(start_times)
        tds = [(t - self.start_time).total_seconds() for t in start_times]
        if max(tds) > 300:
            raise ValueError("Provided channel files have inconsistent start times.")
        self.channels = get_channels(channel_files)
        self.scene = Scene(map(str, channel_files), reader="abi_l1b")

    def extract_inputs(self):
        """
        Extract observations over brasil and return as dictionary mapping
        channel numbers to the corresponding observations.
        """
        inputs = {}
        for c in LOW_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in self.scene.all_dataset_names():
                self.scene.load([f"C{c:02}"])
                x = self.scene[f"C{c:02}"][ROW_START:ROW_END, COL_START:COL_END]
                x = x.load().data.astype(np.float32)
                inputs[c] = x

        for c in MED_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in self.scene.all_dataset_names():
                self.scene.load([f"C{c:02}"])
                x = self.scene[f"C{c:02}"]
                x = x[2 * ROW_START: 2 * ROW_END, 2 * COL_START: 2 * COL_END]
                x = x.load().data.astype(np.float32)
                inputs[c] = x

        hi_res = {}
        for c in HI_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in self.scene.all_dataset_names():
                self.scene.load([f"C{c:02}"])
                x = self.scene[f"C{c:02}"]
                x = x[4 * ROW_START: 4 * ROW_END, 4 * COL_START: 4 * COL_END]
                x = x.load().data.astype(np.float32)
                inputs[c] = x
        return inputs

    def get_retrieval_input(self, normalizer=None):
        """
        Get retrieval input as normalized torch tensors.

        Args:
            normalizer: Tuple of normalizer instances to use
                to normalize the inputs.

        Returns:
            Tuple ``(x_low, x_med, x_high)`` of torch.Tensors containing
            the low-, medium- and high-resolution observations.
        """
        low_res, med_res, hi_res = self.extract_input()
        m = ROW_END - ROW_START
        n = COL_END - COL_START

        inputs = self.extract_inputs()

        low_res = []
        for c in LOW_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in inputs:
                x = inputs[c]
            else:
                x = np.zeros((m, n), dytpe=np.float32)
            low_res.append(x)
        low_res = np.stack(low_res, axis=-1)[np.newaxis]

        for c in MED_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in inputs:
                x = inputs[c]
            else:
                x = np.zeros((2 * m, 2 * n), dytpe=np.float32)
            med_res.append(x)
        med_res = np.stack(med_res, axis=-1)[np.newaxis]

        for c in HI_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in inputs:
                x = inputs[c]
            else:
                x = np.zeros((4 * m, 4 * n), dytpe=np.float32)
            hi_res.append(x)
        hi_res = np.stack(hi_res, axis=-1)[np.newaxis]

        if normalizer is not None:
            low_res = normalizer[0](low_res[np.newaxis])
            med_res = normalizer[0](med_res[np.newaxis])
            hi_res = normalizer[0](hi_res[np.newaxis])

        return (
            torch.tensor(low_res),
            torch.tensor(med_res),
            torch.tensor(hi_res)
        )

    def get_input_data(self):
        """
        Get retrieval input data as xarray Dataset.

        Return:
            xarray.Dataset containing the observations from all available
            channels cropped to a region covering brazil.
        """
        inputs = self.extract_inputs()
        area = self.scene.min_area()[ROW_START:ROW_END, COL_START:COL_END]
        lons, lats = area.get_lonlats()

        start_time = pd.Timestamp(self.start_time).to_datetime64()
        input_data = xr.Dataset({
            "time": (("time",), [start_time]),
            "longitude": (("time", "x", "y"), lons[np.newaxis]),
            "latitude": (("time", "x", "y"), lats[np.newaxis]),
        })

        for c in LOW_RES_CHANNELS:
            if c in inputs:
                input_data[f"C{c:02}"] = (
                    ("time", "x", "y"), inputs[c][np.newaxis]
                )

        for c in MED_RES_CHANNELS:
            if c in inputs:
                input_data[f"C{c:02}"] = (
                    ("time", "x_1000", "y_1000"), inputs[c][np.newaxis]
                )

        for c in HI_RES_CHANNELS:
            if c in inputs:
                input_data[f"C{c:02}"] = (
                    ("time", "x_500", "y_500"), inputs[c][np.newaxis]
                )

        return input_data

    def __repr__(self):
        return f"GOES16File(channels={self.channels})"
