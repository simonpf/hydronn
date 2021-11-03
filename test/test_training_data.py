"""
Tests the loading of training data.
"""
from pathlib import Path
from hydronn.data.training_data import HydronnDataset

def test_training_data():
    """
    Ensure that loading of training data works and that input
    has the expected shape.
    """
    data_path = Path(__file__).parent / "data"
    dataset = HydronnDataset(data_path / "goes_gpm_2017_14.nc",
                             batch_size=8)

    x, y = dataset[0]
    x[0].shape == 4
    x[-1].shape == 128
    x[-2].shape == 128


