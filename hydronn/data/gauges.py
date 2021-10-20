"""
===================
hydronn.data.gauges
===================

An interface to read the gauge data provided by INPE.
"""
from pathlib import Path

import numpy as np
import xarray as xr

DATA_PATH = Path("/home/simonpf/data/hydronn/gauge_data")


def parse_filename(filename):
    """
    Parse information from gauge filename.

    Args:
        filename: Path pointing to a text file containing gauge data.

    Return:
        A tuple ``(state, gauge_id, lon, lat)`` containing the state code
        ``state``, the  auge ID ``id`` as well as the longitude and latitude
        coordinates of the gauge.
    """
    state, gauge_id, _, lon, lat = filename.stem.split("_")
    lon = -float(lon[1:])
    lat = -float(lat[1:])
    return state, gauge_id, lon, lat


def read_data(path):
    """
    Reads the rain gauge dataset into a xarray dataset.

    Returns:
        'xarray.Dataset' containing the measured precipitation for all
        gauges and dates.
    """
    files = list(Path(path).glob("*.csv"))
    sorted(files)

    states = []
    ids = []
    precip_rates = []
    lons = []
    lats = []
    dates = []

    datasets = []


    for f in files:
        print(f)
        dates = np.loadtxt(
            f, dtype=np.datetime64, skiprows=1, usecols=0, delimiter=","
        )
        precip_rates = np.loadtxt(
            f, skiprows=1, usecols=range(1, 25), delimiter=","
        )
        hours = np.timedelta64(1, "h") * np.arange(24)
        dates = dates.reshape(-1, 1) + hours.reshape(1, -1)
        dates = dates.reshape(-1)
        precip_rates = precip_rates.reshape(-1)
        state, gauge_id, lon, lat = parse_filename(f)

        dataset = {
            "time": (("time",), dates),
            "surface_precip": (("time", "gauges"), precip_rates.reshape(-1, 1)),
            "gauges": (("gauges",), [gauge_id]),
            "state": (("gauges",), [state]),
            "longitude": (("gauges",), [lon]),
            "latitude": (("gauges",), [lat])
        }
        datasets.append(xr.Dataset(dataset))

    return xr.concat(datasets, dim="gauges")
