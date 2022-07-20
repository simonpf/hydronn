"""
========================
hydronn.bin.extract_data
========================

This module implements the command line application for
extracting GOES/GPM co-locations.
"""
from concurrent.futures import ProcessPoolExecutor
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from rich.progress import track
import xarray as xr

os.environ["OMP_NUM_THREADS"] = "1"
LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'extract_data' command to top-level CLI.
    """
    parser = subparsers.add_parser(
        'extract_data',
        description='Extract GOES/GPM CMB co-locations.',
        help="Extract training data."
    )
    parser.add_argument(
        'year', metavar='year', type=int,
        help='The year for which to extract co-locations.'
    )
    parser.add_argument(
        'days', metavar='day_1, day_2, ...', type=int, nargs="*",
        help='The day for which to extract co-locations.'
    )
    parser.add_argument(
        'destination', metavar='destination', type=str,
        help='The folder in which to store the extracted co-locations.'
    )
    parser.add_argument('--n_processes',
                        metavar="n",
                        type=int,
                        default=4,
                        help='The number of processes to use for the processing.')
    parser.set_defaults(func=run)


def process_file(gpm_file):
    """
    Process a single gpm_file and return extracted co-locations as
    ``xarray.Dataset``.
    """
    from hydronn.data.gpm import GPMCMBFile
    from hydronn.colocations import extract_colocations
    with TemporaryDirectory() as tmp:
        gpm_file = GPMCMBFile.download(gpm_file, Path(tmp))
        dataset = extract_colocations(gpm_file, tmp)
    return dataset


def run(args):
    """
    This function implements the actual execution of the co-location
    extraction.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from hydronn.data.gpm import get_gpm_files
    year = args.year
    days = args.days

    if not days:
        days = list(range(1, 32))

    gpm_files = []
    for d in days:
        gpm_files = get_gpm_files(year, d)

    if not gpm_files:
        LOGGER.error(
            "No GPM overpasses for year '%s' and day '%s'.",
            year, days
        )

    destination = Path(args.destination)
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)


    pool = ProcessPoolExecutor(max_workers=args.n_processes)
    for d in days:
        tasks = []
        gpm_files = get_gpm_files(year, d)
        for f in gpm_files:
            tasks.append(pool.submit(process_file, f))

        datasets = []
        for t, f in zip(track(tasks, description=f"{year}/{d}"),
                        gpm_files):
            try:
                dataset = t.result()
                if dataset is not None:
                    datasets.append(dataset)
            except Exception as e:
                LOGGER.warning(
                    "Processing of %s  failed with the following "
                    "exception:\n %s", f, e
                )
        if datasets:
            output = destination / f"goes_gpm_{year}_{d:02}.nc"
            dataset = xr.concat(datasets, "scenes", coords="all", join="override")
            dataset.to_netcdf(output)
