"""
Tests the hydronn neural network model.
"""
from pathlib import Path

from hydronn.data.training_data import HydronnDataset
from hydronn.models import Hydronn

def test_hydronn():

    data_path = Path(__file__).parent / "data"
    dataset = HydronnDataset(data_path / "goes_gpm_2017_14.nc",
                             batch_size=1)

    x, y = dataset[0]
    x = [x_i.cpu() for x_i in x]
    y = y.cpu()


    model = Hydronn(128, 2, 128, 4, 128)
    model = model.cpu()
    y_pred = model(x)

    assert y_pred.shape[0] == x[0].shape[0]
    assert y_pred.shape[1] == 128

