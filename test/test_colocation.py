from pathlib import Path

from hydronn.colocations import extract_colocations
from hydronn.data.gpm import GPMCMBFile

def test_extract_colocations():
    data_path = Path(__file__).parent / "data"
    filename = data_path / "2B.GPM.DPRGMI.CORRA2018.20210829-S205206-E222439.042628.V06A.HDF5"
    cache = data_path / "goes"
    extract_colocations(filename, cache)
