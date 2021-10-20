"""
Test interface to read gauge data.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from hydronn.data.gauges import DATA_PATH, parse_filename

def test_parse_filename():
    filename = DATA_PATH / "AC_A102_187_W69.92722221_S9.3575.csv"
    state, gauge_id, lon, lat = parse_filename(filename)
    assert state == "AC"
    assert gauge_id == "A102"
    assert np.isclose(lon, 69.92722221)
    assert np.isclose(lat, 9.3575)




