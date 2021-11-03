from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hydronn.utils import save_and_compress, decompress_and_load


def test_save_and_compress(tmp_path):
    """
    Test reading and saving of gzipped NetCDF files.
    """
    dataset = xr.Dataset({
        "x": (("x",), np.ones(10))
    })
    save_and_compress(dataset, tmp_path / "test.nc")
    assert Path(tmp_path / "test.nc.gz").exists()

    loaded = decompress_and_load(tmp_path / "test.nc")
    assert np.all(loaded.x == dataset.x)

    # Also test loading of uncrompressed data.
    Path(tmp_path / "test.nc.gz").unlink()
    dataset.to_netcdf(tmp_path / "test.nc")
    loaded = decompress_and_load(tmp_path / "test.nc")
    assert np.all(loaded.x == dataset.x)



