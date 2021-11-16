"""
==================================
hydronn.bin.extract_retrieval_data
==================================

This module implements the command line application to download the
retrieval input for a given date range.
"""
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, ThreadPoolExecutor
from datetime import (datetime, timedelta)
import logging
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory, mkdtemp

from rich.progress import track
import xarray as xr

os.environ["OMP_NUM_THREADS"] = "1"
LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'extract_retrieval_data' command to top-level CLI.
    """
    parser = subparsers.add_parser(
        'extract_retrieval_data',
        description='Extract input for the hydronn retrieval.')
    parser.add_argument('year',
                        metavar='year',
                        type=int,
                        help='The year for which to extract inputs.')
    parser.add_argument('month',
                        metavar='month',
                        type=int,
                        help='The month for which to extract inputs.')
    parser.add_argument('days',
                        metavar='day_1, day_2, ...',
                        type=int,
                        nargs="*",
                        help='The day(s) for which to extract inputs.')
    parser.add_argument(
        'destination',
        metavar='destination',
        type=str,
        help='The folder in which to store the extracted data.')
    parser.add_argument(
        '--n_processes',
        metavar="n",
        type=int,
        default=1,
        help='The number of processes to use for the processing.')
    parser.set_defaults(func=run)


def add_channels(datasets):
    """
    Add missing channels to list of datasets.

    Args:
        dataset: List of 'xarray.Dataset' containing GOES input data.

    Return:
        The same list of datasets but with nan values added where some
        channels were missing in some of the datasets.
    """
    import numpy as np
    from hydronn.data.goes import (LOW_RES_CHANNELS, MED_RES_CHANNELS,
                                   HI_RES_CHANNELS, ROW_START, ROW_END,
                                   COL_START, COL_END)
    m = ROW_END - ROW_START
    n = COL_END - COL_START

    for dataset in datasets:
        for i in range(1, 17):
            v = f"C{i:02}"
            if v not in dataset.variables:
                if any([v in d.variables for d in datasets]):
                    if i in LOW_RES_CHANNELS:
                        data = np.zeros((m, n), dtype=np.float32)
                        data[:] = np.nan
                        dataset[v] = (("x", "y"), data)
                    elif i in MED_RES_CHANNELS:
                        data = np.zeros((2 * m, 2 * n), dtype=np.float32)
                        data[:] = np.nan
                        dataset[v] = (("x_1000", "y_1000"), data)
                    elif i in HI_RES_CHANNELS:
                        data = np.zeros((4 * m, 4 * 2), dtype=np.float32)
                        data[:] = np.nan
                        dataset[v] = (("x_500", "y_500"), data)
    return datasets


def process_hour(year, month, day, hour, destination):
    """
    Download and prepare retrieval input for an hour.

    Args:
        year: The year for which to download the data.
        month: The month.
        hour: The day.
        hour: The hour.
        destination: Where to store the results.
    """
    from pansat.download.providers.goes_aws import GOESAWSProvider
    from pansat.products.satellite.goes import (
        goes_16_l1b_radiances_all_full_disk)
    from hydronn.data.goes import GOES16File
    from hydronn.utils import save_and_compress

    filename = (f"hydronn_input_{year:04}_{month:02}"
                f"_{day:02}_{hour:02}.nc")
    output_file = destination / filename
    output_file_c = destination / (str(filename) + ".gz")
    if output_file_c.exists():
        LOGGER.info("File '%s' already exists. Skipping.", output_file_c)
        return None
    else:
        LOGGER.info("File '%s' doesn't exist. Continuing.", output_file_c)

    tmp = mkdtemp()
    tmp = Path(tmp)

    start_time = datetime(year, month, day, hour)
    end_time = start_time + timedelta(hours=1)

    # Download files
    files = []
    provider = GOESAWSProvider(goes_16_l1b_radiances_all_full_disk)
    filenames = provider.get_files_in_range(start_time,
                                            end_time,
                                            start_inclusive=False)
    for f in filenames:
        path = tmp / f
        if not path.exists():
            provider.download_file(f, path)
        files.append(path)

    goes_files = GOES16File.open_files(files)
    datasets = add_channels([f.get_input_data() for f in goes_files])

    # Make sure temporary directory is cleaned up.
    shutil.rmtree(tmp, ignore_errors=True)

    dataset = xr.concat(datasets, dim="time")
    save_and_compress(dataset, output_file)

    del goes_files
    del datasets
    del dataset


def run(args):
    """
    This function implements the actual execution of the co-location
    extraction.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from hydronn.data.gpm import get_gpm_files
    from hydronn.utils import save_and_compress

    logging.basicConfig(level=logging.INFO)

    year = args.year
    month = args.month
    days = args.days
    if not days:
        days = list(range(1, monthrange(year, month) + 1))
    destination = Path(args.destination)
    if not destination.exists():
        destination.mkdir(parents=True)

    pool = ThreadPoolExecutor(max_workers=args.n_processes)
    save_pool = ThreadPoolExecutor(max_workers=1)

    for d in days:
        tasks = []
        hours = list(range(0, 24))
        for h in hours:
            tasks.append(
                pool.submit(process_hour, year, month, d, h, destination))

        for h, t in zip(hours, tasks):
            try:
                data = t.result()
                LOGGER.info("Finished processing hour %s.", h)
            except Exception as e:
                LOGGER.warning(
                    "Processing of hour %s  failed with the following "
                    "exception:\n %s", h, e)
