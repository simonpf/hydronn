"""
=================
hydronn.retrieval
=================

This module handles the processing of the retrieval.
"""
import numpy as np
import xarray as xr
import torch

import quantnn.density as qd

from hydronn.data.goes import (LOW_RES_CHANNELS,
                               MED_RES_CHANNELS,
                               HI_RES_CHANNELS,
                               ROW_START,
                               ROW_END,
                               COL_START,
                               COL_END)
from hydronn.utils import decompress_and_load


class AprioriCorrection:
    """
    The 'AprioriCorrection' class implements a priori correction based
    on Bayes theorem using a precomputed likelihood ratio.
    """
    def __init__(self, ratio_file):
        ratio_data = xr.load_dataset(ratio_file)
        self.n_bins = ratio_data.bins.size
        self.ratios_dep = torch.Tensor(
            ratio_data.ratios_dep.data.astype(np.float16)
        )
        self.ratios_indep = torch.Tensor(
            ratio_data.ratios_indep.data.astype(np.float16)
        )

    def __call__(self, p_dep, p_indep):
        """
        Apply correction to given probabilities.

        Args:
            p_dep: A torch tensor containing probabilities of hourly
                accumulated precipitation assuming dependent errors.
            p_idep: A torch tensor containing probabilities of hourly
                accumulated precipitation assuming independent errors.

        Return:
            A tuple ``(p_dep, p_indep)`` containing the corrected probabilities.
        """
        shape = [1] * p_dep.ndim
        i = min(p_dep.ndim - 1, 1)
        shape[i] = self.n_bins
        ratios_dep = self.ratios_dep.reshape(shape).to(p_dep.device)
        ratios_indep = self.ratios_indep.reshape(shape).to(p_indep.device)
        return (p_dep * ratios_dep, p_indep * ratios_indep)


class InputFile:
    """
    Interface class to load the GOES observations from the NetCDF files
    containing the retrieval input and to prepare the input for the
    neural network model.
    """
    def __init__(self,
                 filename,
                 normalizer,
                 batch_size=-1):
        """
        Args:
            filename: Path to the NetCDF file containing the data.
            normalizer: A normalizer object to use to normalize the
                inputs.
            batch_size: How many observations to combine to a single input.
        """
        self.data = decompress_and_load(filename).sortby("time")
        dims = ["time", "x", "x_500", "x_1000", "y", "y_500", "y_1000"]
        dims = [d for d in dims if d in self.data.dims]
        self.data = self.data.transpose(*dims)
        self.normalizer = normalizer
        self.batch_size = batch_size


    def get_input_data(self, t_start, t_end):
        """
        Get input samples for a range of time indices.

        Args:
            t_start: The starting index of the range of time indices.
            t_end: The next index from the last value in the range of
                time indices.

        Return:
            A 'torch.Tensor' containing the input data for the specified
            range of t indices.
        """
        m = ROW_END - ROW_START
        n = COL_END - COL_START
        t = t_end - t_start

        low_res = []
        for c in LOW_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in self.data:
                x = self.data[channel_name].data[t_start:t_end]
            else:
                x = np.zeros((t, m, n), dtype=np.float32)
                x[:] = np.nan
            low_res.append(x)
        low_res = np.stack(low_res, axis=1)

        med_res = []
        for c in MED_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in self.data:
                x = self.data[channel_name].data[t_start:t_end]
            else:
                x = np.zeros((t, 2 * m, 2 * n), dtype=np.float32)
                x[:] = np.nan
            med_res.append(x)
        med_res = np.stack(med_res, axis=1)

        hi_res = []
        for c in HI_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in self.data:
                x = self.data[channel_name].data[t_start:t_end]
            else:
                x = np.zeros((t, 4 * m, 4 * n), dtype=np.float32)
                x[:] = np.nan
            hi_res.append(x)
        hi_res = np.stack(hi_res, axis=1)

        low_res = self.normalizer[0](low_res)
        med_res = self.normalizer[1](med_res)
        hi_res = self.normalizer[2](hi_res)

        return (
            torch.tensor(low_res),
            torch.tensor(med_res),
            torch.tensor(hi_res)
        )

    def __len__(self):
        """The number of batches in the file."""
        if self.batch_size < 0:
            return 1

        n = self.data.time.size
        return n // self.batch_size + n % self.batch_size

    def __getitem__(self, i):
        """Return a given batch."""
        if (self.batch_size < 0) or (self.batch_size > self.data.time.size):
            return self.get_input_data(0, self.data.time.size)
        else:
            t_start = i * self.batch_size
            t_end = t_start + self.batch_size
            return self.get_input_data(t_start, t_end)

###############################################################################
# Tiling functionality
###############################################################################


def get_start_and_clips(n, tile_size, overlap):
    """
    Calculate start indices and numbers of clipped pixels for a given
    side length, tile size and overlap.

    Args:
        n: The image size to tile in pixels.
        tile_size: The size of each tile
        overlap: The number of pixels of overlap.

    Rerturn:
        A tuple ``(start, clip)`` containing the start indices of each tile
        and the number of pixels to clip between each neighboring tiles.
    """
    start = []
    clip = []
    j = 0
    while j + tile_size < n:
        start.append(j)
        if j > 0:
            clip.append(overlap // 2)
        j = j + tile_size - overlap
    start.append(max(n - tile_size, 0))
    if len(start) > 1:
        clip.append((start[-2] + tile_size - start[-1]) // 2)
    start = start
    clip = clip
    return start, clip


class Tiler:
    """
    Helper class that performs two-dimensional tiling of retrieval inputs and
    calculates clipping ranges for the reassembly of tiled predictions.

    Attributes:
        M: The number of tiles along the first image dimension (rows).
        N: The number of tiles along the second image dimension (columns).
    """
    def __init__(self, x, tile_size=512, overlap=32):
        """
        Args:
            x: List of input tensors for the hydronn retrieval.
            tile_size: The size of a single tile.
            overlap: The overlap between two subsequent tiles.
        """

        self.x = x
        _, _, m, n = x[0].shape
        self.m = m
        self.n = n

        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        if len(tile_size) == 1:
            tile_size = tile_size * 2
        self.tile_size = tile_size
        self.overlap = overlap

        i_start, i_clip = get_start_and_clips(self.m, tile_size[0], overlap)
        self.i_start = i_start
        self.i_clip = i_clip

        j_start, j_clip = get_start_and_clips(self.n, tile_size[1], overlap)
        self.j_start = j_start
        self.j_clip = j_clip

        self.M = len(i_start)
        self.N = len(j_start)

    def get_tile(self, i, j):
        """
        Get tile in the 'i'th row and 'j'th column of the two
        dimensional tiling.

        Args:
            i: The 0-based row index of the tile.
            j: The 0-based column index of the tile.

        Return:
            List containing the tile extracted from the list
            of input tensors.
        """
        i_start = self.i_start[i]
        i_end = i_start + self.tile_size[0]
        j_start = self.j_start[j]
        j_end = j_start + self.tile_size[1]

        x_tile = []
        for x in self.x:
            x_tile.append(x[..., i_start:i_end, j_start:j_end])
            i_start *= 2
            i_end *= 2
            j_start *= 2
            j_end *= 2

        return  x_tile

    def get_slices(self, i, j):
        """
        Return slices for the clipping of the result tensors.

        Args:
            i: The 0-based row index of the tile.
            j: The 0-based column index of the tile.

        Return:
            Tuple of slices that can be used to clip the retrieval
            results to obtain non-overlapping tiles.
        """

        if i == 0:
            i_clip_l = 0
        else:
            i_clip_l = self.i_clip[i - 1]
        if i >= self.M - 1:
            i_clip_r = None
        else:
            i_clip_r = -self.i_clip[i]
        slice_i = slice(i_clip_l, i_clip_r)

        if j == 0:
            j_clip_l = 0
        else:
            j_clip_l = self.j_clip[j - 1]
        if j >= self.N - 1:
            j_clip_r = None
        else:
            j_clip_r = -self.j_clip[j]
        slice_j = slice(j_clip_l, j_clip_r)

        return (slice_i, slice_j)


    def __repr__(self):
        return f"Tiler(tile_size={self.tile_size}, overlap={self.overlap})"


class Retrieval:
    """
    Processor class to run the retrieval for a list of input files.
    """
    def __init__(self,
                 input_files,
                 model,
                 normalizer,
                 tile_size=256,
                 overlap=32,
                 device="cuda",
                 correction=None):
        """
        Args:
            input_files: The list of input files for which to run the
                retrieval.
            model: The model to use for the retrieval.
            normalizer: The normalizer object to use to normalize the
                inputs.
            tile_size: The sizes of the tiles for the splitting of
                the input domain.
            device: The device on which to run the retrieval.
        """
        self.input_files = sorted(input_files)
        self.model = model
        self.normalizer = normalizer
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = device
        if correction is not None:
            self.correction = AprioriCorrection(correction)
        else:
            self.correction = None

    def _run_file(self, input_file):
        """
        This function implements the processing of observations for one hour.
        For each input in the input file the posterior distributions are
        calculated and accumulated to hourly predictions.

        Args:
            input_file: InputFile object providing access to the observations
                for a given hour.

        Return:
            An 'xarray.Dataset' containing the retrieval results.
        """
        input_data = InputFile(input_file, self.normalizer, batch_size=6)
        quantiles = [
            0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
            0.7, 0.8, 0.9, 0.95, 0.99, 0.999
        ]
        device = self.device

        bins = torch.Tensor(self.model.bins).to(device)
        d_bins = bins[1:] - bins[:-1]

        model = self.model.model.to(device)

        x = input_data[0]
        tiler = Tiler(x, tile_size=self.tile_size, overlap=self.overlap)

        sample_dep = []
        quantiles_dep = []
        mean_dep = []
        sample_dep_c = []
        quantiles_dep_c = []
        mean_dep_c = []

        sample_indep = []
        quantiles_indep = []
        mean_indep = []
        sample_indep_c = []
        quantiles_indep_c = []
        mean_indep_c = []

        for i in range(tiler.M):

            sample_dep.append([])
            quantiles_dep.append([])
            mean_dep.append([])

            sample_indep.append([])
            quantiles_indep.append([])
            mean_indep.append([])

            if self.correction is not None:
                sample_dep_c.append([])
                quantiles_dep_c.append([])
                mean_dep_c.append([])

                sample_indep_c.append([])
                quantiles_indep_c.append([])
                mean_indep_c.append([])

            for j in range(tiler.N):


                with torch.no_grad():
                    # Retrieve tile
                    x_t = tiler.get_tile(i, j)
                    slices = tiler.get_slices(i, j)

                    y_pred_dep = None
                    y_pred_indep = None

                    # Process each sample in batch separately.
                    n_inputs = x_t[0].shape[0]
                    bins_acc = (1 / n_inputs) * bins

                    for k in range(x_t[0].shape[0]):
                        x_t_b = [t[[k]] for t in x_t]
                        x_t_b = [t.to(device) for t in x_t_b]
                        y_pred = model(x_t_b)[(...,) + slices]
                        y_pred = self.model._post_process_prediction(
                            y_pred,
                            bins
                        )

                        if y_pred_dep is None:
                            y_pred_dep = (1 / n_inputs) * y_pred
                            y_pred_indep = y_pred
                        else:
                            y_pred_dep = y_pred_dep + (1 / n_inputs) * y_pred
                            y_pred_indep = qd.add(
                                y_pred_indep, bins, y_pred, bins, bins
                            )

                    sample_dep[-1].append(qd.sample_posterior(
                        y_pred_dep, bins
                    ).cpu().numpy()[:, 0])
                    quantiles_dep[-1].append(qd.posterior_quantiles(
                        y_pred_dep, bins, quantiles
                    ).cpu().numpy().transpose([0, 2, 3, 1]))
                    mean_dep[-1].append(qd.posterior_mean(
                        y_pred_dep, bins
                    ).cpu().numpy())

                    y_pred_indep = n_inputs * y_pred_indep
                    sample_indep[-1].append(qd.sample_posterior(
                        y_pred_indep, bins_acc
                    ).cpu().numpy()[:, 0])
                    quantiles_indep[-1].append(qd.posterior_quantiles(
                        y_pred_indep, bins_acc, quantiles
                    ).cpu().numpy().transpose([0, 2, 3, 1]))
                    mean_indep[-1].append(qd.posterior_mean(
                        y_pred_indep, bins_acc
                    ).cpu().numpy())

                    if self.correction:
                        y_pred_dep_c, y_pred_indep_c = self.correction(
                            y_pred_dep,
                            y_pred_indep
                        )
                        y_pred_dep_c = qd.normalize(y_pred_dep_c, bins, 1, True)
                        y_pred_indep_c = qd.normalize(y_pred_indep_c, bins, 1, True)
                        sample_dep_c[-1].append(qd.sample_posterior(
                            y_pred_dep_c, bins
                        ).cpu().numpy()[:, 0])
                        quantiles_dep_c[-1].append(qd.posterior_quantiles(
                            y_pred_dep_c, bins, quantiles
                        ).cpu().numpy().transpose([0, 2, 3, 1]))
                        mean_dep_c[-1].append(qd.posterior_mean(
                            y_pred_dep_c, bins
                        ).cpu().numpy())

                        sample_indep_c[-1].append(qd.sample_posterior(
                            y_pred_indep_c, bins_acc
                        ).cpu().numpy()[:, 0])
                        quantiles_indep_c[-1].append(qd.posterior_quantiles(
                            y_pred_indep_c, bins_acc, quantiles
                        ).cpu().numpy().transpose([0, 2, 3, 1]))
                        mean_indep_c[-1].append(qd.posterior_mean(
                            y_pred_indep_c, bins_acc
                        ).cpu().numpy())

        # Finally, concatenate over rows and columns.
        sample_dep = np.concatenate(
            [np.concatenate(r, -1) for r in sample_dep], -2)
        quantiles_dep = np.concatenate(
            [np.concatenate(r, -2) for r in quantiles_dep], -3
        )
        mean_dep = np.concatenate(
            [np.concatenate(r, -1) for r in mean_dep], -2
        )

        sample_indep = np.concatenate(
            [np.concatenate(r, -1) for r in sample_indep], -2
        )
        quantiles_indep = np.concatenate(
            [np.concatenate(r, -2) for r in quantiles_indep], -3
        )
        mean_indep = np.concatenate(
            [np.concatenate(r, -1) for r in mean_indep], -2
        )

        dims = ("time", "x", "y")
        results = xr.Dataset({
            "time":  ("time", input_data.data.time.mean("time").data.reshape((1,))),
            "n_inputs": (("time",), [n_inputs]),
            "latitude": input_data.data.latitude.mean("time"),
            "longitude": input_data.data.longitude.mean("time"),
            "quantiles": (("quantiles",), quantiles),
            "mean_dep": (dims, mean_dep),
            "sample_dep": (dims, sample_dep),
            "quantiles_dep": (dims + ("quantiles",), quantiles_dep),
            "mean_indep": (dims, mean_indep),
            "sample_indep": (dims, sample_indep),
            "quantiles_indep": (dims + ("quantiles",), quantiles_indep)
        })

        if self.correction:
            # Finally, concatenate over rows and columns.
            sample_dep_c = np.concatenate(
                [np.concatenate(r, -1) for r in sample_dep_c], -2)
            quantiles_dep_c = np.concatenate(
                [np.concatenate(r, -2) for r in quantiles_dep_c], -3
            )
            mean_dep_c = np.concatenate(
                [np.concatenate(r, -1) for r in mean_dep_c], -2
            )

            sample_indep_c = np.concatenate(
                [np.concatenate(r, -1) for r in sample_indep_c], -2
            )
            quantiles_indep_c = np.concatenate(
                [np.concatenate(r, -2) for r in quantiles_indep_c], -3
            )
            mean_indep_c = np.concatenate(
                [np.concatenate(r, -1) for r in mean_indep_c], -2
            )

            results["mean_dep_c"] = (dims, mean_dep_c)
            results["sample_dep_c"] = (dims, sample_dep_c)
            results["quantiles_dep_c"] = (dims + ("quantiles",), quantiles_dep_c)
            results["mean_indep_c"] = (dims, mean_indep_c)
            results["sample_indep_c"] = (dims, sample_indep_c)
            results["quantiles_indep_c"] = (dims + ("quantiles",), quantiles_indep_c)

        return results

    def run(self):
        """
        Run retrieval and return results as 'xarray.Dataset'.
        """
        results = []
        for f in self.input_files:
            results.append(self._run_file(f))
        return xr.concat(results, "time")
