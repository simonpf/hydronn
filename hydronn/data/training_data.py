
import xarray as xr
import numpy as np

from quantnn.normalizer import MinMaxNormalizer

class HydronnDataset:
    def __init__(self,
                 filename,
                 batch_size,
                 shuffle=True,
                 normalize=True,
                 normalizer=None):
        self.filename = filename
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.load_data()

        # Normalize input data if necessary.
        if normalize:
            if normalizer is None:
                normalizer = tuple([
                    MinMaxNormalizer(x) for x in [
                        self.low_res,
                        self.med_res,
                        self.hi_res
                    ]
                ])
            self.normalizer = normalizer
            self.low_res = self.normalizer[0](self.low_res)
            self.med_res = self.normalizer[0](self.med_res)
            self.hi_res = self.normalizer[0](self.hi_res)

        self.shuffle = shuffle
        self.shuffled = False
        if shuffle:
            self._shuffle()

        self._replace_zeros()

    def _shuffle(self):
        if not self.shuffled:
            indices = np.random.permutation(self.low_res.shape[0])
            self.low_res = self.low_res[indices]
            self.med_res = self.med_res[indices]
            self.hi_res = self.hi_res[indices]
            self.surface_preicp = self.surface_precip[indices]
            self.shuffled = True

    def _replace_zeros(self):
        indices = self.surface_precip < 1e-2
        new = 10 ** np.random.uniform(-4, -2, shape=indices.size)
        self.surface_precip[indices] = new

    def load_data(self):
        with xr.open_dataset(self.filename) as data:
            hi_res = data["C01"].data[:, np.newaxis]
            med_res = np.stack([
                data["C01"].data,
                data["C03"].data,
                data["C05"].data,
                ], axis=1)
            low_res = np.stack([
                data["C{i:02}"] for i in [4] + range(6, 17)
            ])
            surface_precip = data.surface_precip.data
            surface_precip = np.nan_to_num(surface_precip, nan=-1)

        self.hi_res = hi_res
        self.med_res = med_res
        self.low_res = low_res
        self.surface_precip = surface_precip
