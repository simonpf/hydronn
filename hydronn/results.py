"""
===============
hydronn.results
===============

Functions for the post processing of the retrieval results.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import xarray as xr

from rich.progress import track
import numpy as np

from hydronn.utils import decompress_and_load

DATA_PATH = Path(__file__).parent / "files"
GAUGE_COORDS = xr.open_dataset(DATA_PATH / "gauge_coordinates.nc")


def process_file(filename):
    """
    Helper function to interpolate retrieval results to gauge
    coordinates

    Args:
        filename: Path to the retrieval function to process.

    Return:
        'xarray.Datset' containing the retrieval results interpolated to
        the Gauge coordinates.
    """
    data = decompress_and_load(filename)
    if "x_4" in data.dims:
        x = GAUGE_COORDS.x
        y = GAUGE_COORDS.y
        x_4 = GAUGE_COORDS.x_4
        y_4 = GAUGE_COORDS.y_4
        return data.interp(
            {
                "x": x,
                "y": y,
                "x_4": x_4,
                "y_4": y_4,
            }
        )
    else:
        x = GAUGE_COORDS.x
        y = GAUGE_COORDS.y
        return data.interp({"x": x, "y": y})


def interpolate_results(result_path):
    """
    Interpolate hydronn retrieval results to gauge locations.

    This function tries to read all files with suffix ".nc.gz" in the
    given directory tree, interpolates them to the location of the
    gauges and concatenates the results into a single dataset.

    Args:
        results_path: The path in which to look for result files.

    Return:
        'xarray.Dataset' containing the retrieval results interpolated
        to the gauge locations.
    """
    result_path = Path(result_path)

    files = sorted(list(result_path.glob("**/*.nc.gz")))
    results = []

    pool = ProcessPoolExecutor(max_workers=6)

    tasks = [pool.submit(process_file, f) for f in files]
    for f in files:
        pool.submit(process_file, f)

    for t, f in track(list(zip(tasks, files))):
        data = t.result()
        print(f"Done processing {f}.")
        if "n_inputs" in data.variables:
            data = data.drop("n_inputs")
        results.append(data)

    return xr.concat(results, "time")


def split_list(list_in, n):
    """
    Split list into n parts.
    """
    l_sub = len(list_in) // n
    modulus = len(list_in) % n
    lists_out = []
    i = 0
    while list_in:
        if i < modulus:
            l = l_sub + 1
        else:
            l = l_sub
        part, list_in = list_in[:l], list_in[l:]
        lists_out.append(part)
        i += 1
    return lists_out


def process_files(filenames):
    """
    Load retrieval results from a list of file and accumulate results.
    """
    n = 0
    results = None
    variables = [
        "mean_dep",
        "sample_dep",
        "mean_dep_c",
        "sample_dep_c",
        "mean_indep",
        "sample_indep",
        "mean_indep_c",
        "sample_indep_c",
    ]

    for filename in filenames:
        data = decompress_and_load(filename)
        if results is None:
            results = data[variables]
        else:
            for variable in variables:
                results[variable].data += data[variable].data
        n += 1
        print(f"Done processing {filename}.")
    results.attrs["files"] = n
    return results


def calculate_accumulations(result_path, start=None, end=None):
    """
    Interpolate hydronn retrieval results to gauge locations.

    This function tries to read all files with suffix ".nc.gz" in the
    given directory tree and to  calculate precipitation acummulation
    and averages.

    Args:
        results_path: The path in which to look for result files.
        start: Optional numpy.datetime64 specifying start of a time
            interval over which to accumulate the precipitation.
        end: Optional numpy.datetime64 specifying end of a time
            interval over which to accumulate the precipitation.

    Return:
        'xarray.Dataset' containing the accumulated and average precipitation.
    """
    result_path = Path(result_path)

    all_files = sorted(list(result_path.glob("**/*.nc.gz")))
    files = []
    for f in all_files:
        if start is not None and end is not None:
            parts = f.name.split(".")[0].split("_")
            year = parts[2]
            month = parts[3]
            day = parts[4]
            hour = parts[5]
            date = np.datetime64(f"{year}-{month}-{day}T{hour}:00:00")
            print(date)
            if date < start or date >= end:
                continue
        files.append(f)

    file_lists = split_list(files, 6)

    pool = ProcessPoolExecutor(max_workers=12)
    tasks = [pool.submit(process_files, f) for f in file_lists]

    variables = [
        "mean_dep",
        "sample_dep",
        "mean_dep_c",
        "sample_dep_c",
        "mean_indep",
        "sample_indep",
        "mean_indep_c",
        "sample_indep_c",
    ]

    result = None
    for t, f in track(list(zip(tasks, files))):
        data = t.result()
        if result is None:
            result = data[variables]

        else:
            for variable in variables:
                result[variable].data += data[variable].data
            result.attrs["files"] += data.attrs["files"]

    # Add latitude and longitude coordinates.
    data = decompress_and_load(files[0])
    lats = data.latitude.data[0]
    lons = data.longitude.data[0]

    if "x_4" in result.dims:
        lats = 0.25 * (
            lats[0::2, 0::2] + lats[0::2, 1::2] + lats[1::2, 0::2] + lats[1::2, 1::2]
        )
        lons = 0.25 * (
            lons[0::2, 0::2] + lons[0::2, 1::2] + lons[1::2, 0::2] + lons[1::2, 1::2]
        )
        result["latitude"] = (("x_4", "y_4"), lats)
        result["longitude"] = (("x_4", "y_4"), lons)
    else:
        result["latitude"] = (("x", "y"), lats)
        result["longitude"] = (("x", "y"), lons)

    return result
