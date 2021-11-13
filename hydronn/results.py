"""
===============
hydronn.results
===============

Functions for the post processing of the retrieval results.
"""
from pathlib import Path

import xarray as xr

from hydronn.utils import decompress_and_load


DATA_PATH = Path(__file__).parent / "files"

GAUGE_COORDS = xr.open_dataset(DATA_PATH / "gauge_coordinates.nc")


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
    x = GAUGE_COORDS.x
    y = GAUGE_COORDS.y

    files = sorted(list(result_path.glob("**/*.nc.gz")))
    results = []
    for f in files[:10]:
        data = decompress_and_load(f)
        results.append(data.interp({"x": x, "y": y}))
        print(f"Done processing {f}.")
    return xr.concat(results, "time")

