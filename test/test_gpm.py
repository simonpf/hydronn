"""
Tests for the 'hydronn.data.gpm' module.
"""
from pathlib import Path

import numpy as np

from hydronn.data.gpm import GPMCMBFile


def test_to_xarray_dataset():
    """
    Test reading of GPM combined data and extraction of data within
    given ROI.
    """
    filename = Path(__file__).parent / "data"
    filename = filename / "2B.GPM.DPRGMI.CORRA2018.20210829-S205206-E222439.042628.V06A.HDF5"
    roi = [-85, -40, -30, 10]
    data = GPMCMBFile(filename).to_xarray_dataset(roi=roi)

    lons = data.longitude.data
    lats = data.latitude.data

    assert np.all(np.any((lats > roi[1]) * (lats < roi[3]), axis=-1))
    assert np.all(np.any((lons > roi[0]) * (lons < roi[2]), axis=-1))


