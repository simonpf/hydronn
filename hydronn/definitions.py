"""
===================
hydronn.definitions
===================

Defines constants used throughout the 'hydronn' package.
"""
from pathlib import Path


# Lower left and upper right corners of the bounding box defining
# the region used to collect training data.
ROI = [-85, -40, -30, 10]

HYDRONN_DATA_PATH = Path(__file__).parent.parent / "data"

_area_file = Path(__file__).parent / "files" / "brazil.yml"

try:
    from pyresample.area_config import load_area_from_string
    BRAZIL = load_area_from_string(open(_area_file, "r").read())
except ModuleNotFoundError:
    pass
