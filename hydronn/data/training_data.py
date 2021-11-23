"""
==========================
hydronn.data.training_data
==========================

Defines a dataset class to load the training data.
"""
import random

import numpy as np
import torch
import xarray as xr

from quantnn.normalizer import MinMaxNormalizer

from hydronn.utils import decompress_and_load


class HydronnDataset:
    """
    Training dataset consisting of GOES16 observations and co-located surface
    precipitation from GPM combined retrievals.
    """
    def __init__(self,
                 filename,
                 batch_size=4,
                 shuffle=True,
                 normalize=True,
                 normalizer=None,
                 augment=True,
                 resolution=2):
        """
        Load training data.

        Args:
            filename: The NetCDF4 file containing the training data.
            batch_size: How many scenes to combined into one batch.
            shuffle: Whether or not to shuffle the data after every epoch.
            normalize: Whether or not to normalize the input data.
            normalizer: Tuple of normalizers to use for the low-, medium- and
                high-resolution input data.
            augment: Whether or not to augment the input data with random
                flips.
            resolution: 'int' specifying the resolution in km of the surface
                precipitation ('2' or '4').
        """
        self.filename = filename
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.resolution = resolution

        seed = random.SystemRandom().randint(0, 2 ** 16)
        self.rng = np.random.default_rng(seed)

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
            self.med_res = self.normalizer[1](self.med_res)
            self.hi_res = self.normalizer[2](self.hi_res)

        self.shuffle = shuffle
        self.shuffled = False
        if shuffle:
            self._shuffle()

        self._replace_zeros()

    def _shuffle(self):
        """
        Shuffle samples in dataset.
        """
        if not self.shuffled:
            indices = self.rng.permutation(self.low_res.shape[0])
            self.low_res = self.low_res[indices]
            self.med_res = self.med_res[indices]
            self.hi_res = self.hi_res[indices]
            self.surface_precip = self.surface_precip[indices]
            self.shuffled = True

    def _replace_zeros(self):
        """
        Replaces all surface precipitation values below 0.01 with small
        random values. This is so that predicted quantiles can be well
        calibrated.
        """
        indices = (self.surface_precip < 1e-2) * (self.surface_precip >= 0)
        shape = self.surface_precip[indices].shape
        new = 10 ** np.random.uniform(-4, -2, size=shape)
        self.surface_precip[indices] = new

    def load_data(self):
        """
        Load input observations and surface precipitation from file.
        Also augments the training samples if the objects 'augment'
        attribute is set to 'True'.
        """
        data = decompress_and_load(self.filename)
        hi_res = data["C02"].data[:, np.newaxis].astype(np.float32)
        med_res = np.stack([
            data["C01"].data.astype(np.float32),
            data["C03"].data.astype(np.float32),
            data["C05"].data.astype(np.float32),
            ], axis=1)
        low_res = np.stack([
            data[f"C{i:02}"].astype(np.float32)
            for i in [4] + list(range(6, 17))
        ], axis=1)
        surface_precip = data.surface_precip.data.astype(np.float32)
        surface_precip = np.nan_to_num(surface_precip, nan=-1)
        if self.resolution > 2:
            surface_precip = surface_precip[:, ::2, ::2]

        data.close()

        self.hi_res = hi_res
        self.med_res = med_res
        self.low_res = low_res
        self.surface_precip = surface_precip

        if self.augment:
            n_scenes = self.hi_res.shape[0]

            indices = self.rng.random(n_scenes) > 0.5
            self.low_res[indices] = np.flip(self.low_res[indices], -2)
            self.med_res[indices] = np.flip(self.med_res[indices], -2)
            self.hi_res[indices] = np.flip(self.hi_res[indices], -2)
            self.surface_precip[indices] = np.flip(self.surface_precip[indices], -2)

            indices = self.rng.random(n_scenes) > 0.5
            self.low_res[indices] = np.flip(self.low_res[indices], -1)
            self.med_res[indices] = np.flip(self.med_res[indices], -1)
            self.hi_res[indices] = np.flip(self.hi_res[indices], -1)
            self.surface_precip[indices] = np.flip(self.surface_precip[indices], -1)

            indices = self.rng.random(n_scenes) > 0.5
            self.low_res[indices] = np.transpose(self.low_res[indices],
                                                 [0, 1, 3, 2])
            self.med_res[indices] = np.transpose(self.med_res[indices],
                                                 [0, 1, 3, 2])
            self.hi_res[indices] = np.transpose(self.hi_res[indices],
                                                [0, 1, 3, 2])
            self.surface_precip[indices] = np.transpose(self.surface_precip[indices],
                                                        [0, 2, 1])

    def __len__(self):
        n_scenes = self.low_res.shape[0]
        n = n_scenes // self.batch_size
        if n_scenes % self.batch_size:
            n += 1
        return n

    def __getitem__(self, i):
        """
        Return batch from training data.
        """
        if i == 0 and self.shuffle:
            self._shuffle()
        self.shuffled = False

        i_start = i * self.batch_size
        i_end = i_start + self.batch_size

        low_res = self.low_res[i_start:i_end]
        med_res = self.med_res[i_start:i_end]
        hi_res = self.hi_res[i_start:i_end]
        x = (
            torch.tensor(low_res),
            torch.tensor(med_res),
            torch.tensor(hi_res),
        )

        precip = self.surface_precip[i_start:i_end]
        y = torch.tensor(precip)
        return (x, y)

