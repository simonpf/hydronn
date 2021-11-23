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

from hydronn.utils import decompress_and_load


DATA_PATH = Path(__file__).parent / "files"

GAUGE_COORDS = xr.open_dataset(DATA_PATH / "gauge_coordinates.nc")


def process_file(filename):
    x = GAUGE_COORDS.x
    y = GAUGE_COORDS.y
    data = decompress_and_load(filename)
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

    pool = ProcessPoolExecutor(max_workers=4)

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

