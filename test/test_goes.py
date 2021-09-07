"""
Tests for the hydronn.data.goes module.
"""
from datetime import datetime
from pathlib import Path

from hydronn.data.goes import (find_goes_16_l1b_files,
                               get_start_times,
                               find_start_times,
                               GOES16File)


def test_find_files():
    """
    Ensure that all 17 files in the 'data/goes' folder are found.
    """
    path = Path(__file__).parent / "data" / "goes"
    files = find_goes_16_l1b_files(path)
    assert len(files) == 17


def test_get_start_times():
    """
    Ensure that the two discrete start times are correctly identified.
    """
    path = Path(__file__).parent / "data" / "goes"
    files = find_goes_16_l1b_files(path)
    start_times = get_start_times(files)
    assert len(start_times) == 2
    assert all([t.year == 2021 for t in start_times])


def test_find_start_times():
    """
    Ensure that the two discrete start times are correctly identified.
    """
    path = Path(__file__).parent / "data" / "goes"
    start_times = find_start_times(path)
    assert len(start_times) == 2
    assert all([t.year == 2021 for t in start_times])


def test_download():
    """
    Assert that download function correctly loads the 16 channels files
    from cache.
    """
    cache = Path(__file__).parent / "data" / "goes"
    time = datetime.strptime("2021241220020", "%Y%j%H%M%S")
    file = GOES16File.download(time, cache)
    assert len(file.channels) == 16


def test_open_files():
    """
    Ensure that 'open_files' method correctly combines the files in the
    test data into two separate files.
    """
    path = Path(__file__).parent / "data" / "goes"
    files = find_goes_16_l1b_files(path)
    files = GOES16File.open_files(files)
    assert len(files) == 2

def test_open_path():
    """
    Ensure that 'open_path' method correctly opens files in the test data
    folder.
    """
    path = Path(__file__).parent / "data" / "goes"
    files = GOES16File.open_path(path)
    assert len(files) == 2

