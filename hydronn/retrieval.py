"""
=================
hydronn.retrieval
=================

This module contains the implementation of the hydronn retrieval that
produces hourly accumulations from GOES input observations.
"""
import numpy as np
import xarray as xr
import torch

import quantnn.density as qd

from hydronn.data.goes import (
    LOW_RES_CHANNELS,
    MED_RES_CHANNELS,
    HI_RES_CHANNELS,
    ROW_START,
    ROW_END,
    COL_START,
    COL_END,
)
from hydronn.utils import decompress_and_load
from hydronn.data.training_data import HydronnDataset

######################################################################
# Correction
######################################################################


class AprioriCorrection:
    """
    The 'AprioriCorrection' class implements a priori correction based
    on  a precomputed likelihood ratio.
    """

    def __init__(self, ratio_file):
        """
        Load a priori correction.

        Args:
            ratio_file: File path pointing to a correction file.
        """
        ratio_data = xr.load_dataset(ratio_file)
        self.n_bins = ratio_data.bins.size
        self.ratios_dep = torch.Tensor(ratio_data.ratios_dep.data.astype(np.float32))
        self.ratios_indep = torch.Tensor(
            ratio_data.ratios_indep.data.astype(np.float32)
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


######################################################################
# Retrieval input
######################################################################


class InputFile:
    """
    Interface class to load the GOES observations from the NetCDF files
    containing the retrieval input and to prepare the input for the
    neural network model.
    """

    def __init__(self, filename, normalizer, batch_size=-1):
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
        self._load_data()

    def _load_data(self):
        """
        Load input data from file. Data is loaded into the 'x' attribute of
        the object.
        """
        m = ROW_END - ROW_START
        n = COL_END - COL_START
        t = self.data.time.size

        low_res = []
        for c in LOW_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in self.data:
                x = self.data[channel_name].data
            else:
                x = np.zeros((t, m, n), dtype=np.float32)
                x[:] = np.nan
            low_res.append(x)
        low_res = np.stack(low_res, axis=1)

        med_res = []
        for c in MED_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in self.data:
                x = self.data[channel_name].data
            else:
                x = np.zeros((t, 2 * m, 2 * n), dtype=np.float32)
                x[:] = np.nan
            med_res.append(x)
        med_res = np.stack(med_res, axis=1)

        hi_res = []
        for c in HI_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in self.data:
                x = self.data[channel_name].data
            else:
                x = np.zeros((t, 4 * m, 4 * n), dtype=np.float32)
                x[:] = np.nan
            hi_res.append(x)
        hi_res = np.stack(hi_res, axis=1)

        invalid = np.any(np.all(np.isnan(low_res), axis=1), axis=(-2, -1))
        invalid *= np.any(np.all(np.isnan(med_res), axis=1), axis=(-2, -1))
        invalid *= np.any(np.all(np.isnan(hi_res), axis=1), axis=(-2, -1))
        valid = ~invalid
        low_res = low_res[valid]
        med_res = med_res[valid]
        hi_res = hi_res[valid]

        low_res = self.normalizer[0](low_res)
        med_res = self.normalizer[1](med_res)
        hi_res = self.normalizer[2](hi_res)

        self.x = (torch.tensor(low_res), torch.tensor(med_res), torch.tensor(hi_res))

    def __len__(self):
        """The number of batches in the file."""
        if self.batch_size < 0:
            return 1

        n = self.data.time.size
        return n // self.batch_size + n % self.batch_size

    def __getitem__(self, i):
        """Return a given batch."""
        if (self.batch_size < 0) or (self.batch_size > self.data.time.size):
            return self.x
        else:
            t_start = i * self.batch_size
            t_end = t_start + self.batch_size
            return [t[t_start:t_end] for t in self.x]


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

    def __init__(self, x, tile_size=512, overlap=32, resolution=2):
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

        self.resolution = resolution

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

        return x_tile

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
        scaling = 1
        if self.resolution > 2:
            scaling = 2

        if i == 0:
            i_clip_l = 0
        else:
            i_clip_l = self.i_clip[i - 1] // scaling
        if i >= self.M - 1:
            i_clip_r = self.tile_size[0]
        else:
            i_clip_r = self.tile_size[0] - self.i_clip[i] // scaling
        slice_i = slice(i_clip_l, i_clip_r)

        if j == 0:
            j_clip_l = 0
        else:
            j_clip_l = self.j_clip[j - 1] // scaling
        if j >= self.N - 1:
            j_clip_r = self.tile_size[1]
        else:
            j_clip_r = self.tile_size[0] - self.j_clip[j] // scaling
        slice_j = slice(j_clip_l, j_clip_r)

        return (slice_i, slice_j)

    def get_weights(self, i, j):
        """
        Get weights to reassemble results.

        Args:
            i: Row-index of the tile.
            j: Column-index of the tile.

        Return:
            Numpy array containing weights for the corresponding tile.
        """
        scaling = 1
        if self.resolution > 2:
            scaling = 2

        sl_i, sl_j = self.get_slices(i, j)

        m, n  = self.tile_size
        w_i = np.ones((m // scaling, n // scaling))
        if i > 0:
            trans_start = self.i_start[i]
            trans_end = self.i_start[i - 1] + self.tile_size[0]
            l_trans = trans_end - trans_start
            start = l_trans // scaling
            w_i[:start] = np.linspace(0, 1, l_trans // scaling)[..., np.newaxis]
        if i < self.M - 1:
            trans_start = self.i_start[i + 1]
            trans_end = self.i_start[i] + self.tile_size[0]
            l_trans = trans_end - trans_start
            start = (self.tile_size[0] - l_trans) // scaling
            w_i[start:] = np.linspace(1, 0, l_trans // scaling)[..., np.newaxis]

        w_j = np.ones((m // scaling, n // scaling))
        if j > 0:
            trans_start = self.j_start[j]
            trans_end = self.j_start[j - 1] + self.tile_size[1]
            l_trans = trans_end - trans_start
            start = l_trans // scaling
            w_j[:, :start] = np.linspace(0, 1, l_trans // scaling)[np.newaxis]
        if j < self.N - 1:
            trans_start = self.j_start[j + 1]
            trans_end = self.j_start[j] + self.tile_size[1]
            l_trans = trans_end - trans_start
            start = (self.tile_size[1] - l_trans) // scaling
            w_j[:, start:] = np.linspace(1, 0, l_trans // scaling)[np.newaxis]

        return w_i * w_j

    def assemble(self, slices):
        """
        Assemble slices back to original shape using linear interpolation in
        overlap regions.

        Args:
            slices: List of lists of slices.

        Return:
            ``numpy.ndarray`` containing the data from the slices reconstructed
            to the original shape.
        """
        slice_0 = slices[0][0]
        scaling = 1
        if self.resolution > 2:
            scaling = 2

        shape = slice_0.shape[:-2] + (self.m // scaling, self.n // scaling)
        results = np.zeros(shape, dtype=slice_0.dtype)

        for i, row in enumerate(slices):
            for j, slc in enumerate(row):

                i_start = self.i_start[i]
                i_end = i_start + self.tile_size[0]
                row_slice = slice(i_start // scaling, i_end // scaling)
                j_start = self.j_start[j]
                j_end = j_start + self.tile_size[1]
                col_slice = slice(j_start // scaling, j_end // scaling)

                output = results[..., row_slice, col_slice]
                weights = self.get_weights(i, j)
                output += weights * slc

        return results

    def __repr__(self):
        return f"Tiler(tile_size={self.tile_size}, overlap={self.overlap})"


###############################################################################
# The retrieval
###############################################################################


class Retrieval:
    """
    Processor class to run the retrieval for a list of input files.
    """

    def __init__(
        self,
        input_files,
        model,
        normalizer,
        tile_size=256,
        overlap=32,
        device="cuda",
        correction=None,
    ):
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
        self.input_files = input_files
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
        if not isinstance(input_file, InputFile):
            input_data = InputFile(input_file, self.normalizer, batch_size=6)
        else:
            input_data = input_file
        quantiles = [
            0.01,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.99,
            0.999,
        ]
        device = self.device

        bins = torch.Tensor(self.model.bins).to(device)
        d_bins = bins[1:] - bins[:-1]

        model = self.model.model.to(device)

        x = input_data[0]
        tiler = Tiler(
            x,
            tile_size=self.tile_size,
            overlap=self.overlap,
            resolution=self.model.resolution,
        )

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

        any_invalid = False
        any_ir_invalid = False

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

                    y_pred_dep = None
                    y_pred_indep = None

                    # Process each sample in batch separately.
                    n_inputs = x_t[0].shape[0]
                    bins_acc = (1 / n_inputs) * bins

                    for k in range(x_t[0].shape[0]):
                        x_t_b = [t[[k]] for t in x_t]

                        # Check
                        low_res, med_res, hi_res = x
                        any_invalid += ~(
                            np.any(np.isfinite(hi_res.numpy()), (-3, -2, -1)) +
                            np.any(np.isfinite(med_res.numpy()), (-3, -2, -1)) +
                            np.any(np.isfinite(low_res.numpy()), (-3, -2, -1))
                        )
                        any_ir_invalid += ~np.any(
                            np.isfinite(low_res.numpy()[:, -4]), (-2, -1)
                        )

                        x_t_b = [t.to(device) for t in x_t_b]
                        y_pred = model(x_t_b)
                        y_pred = self.model._post_process_prediction(y_pred, bins)

                        if y_pred_dep is None:
                            y_pred_dep = (1 / n_inputs) * y_pred
                            y_pred_indep = y_pred
                        else:
                            y_pred_dep = y_pred_dep + (1 / n_inputs) * y_pred
                            y_pred_indep = qd.add(
                                y_pred_indep / (k / (k + 1)),
                                bins * (k / (k + 1)),
                                y_pred * (k + 1),
                                bins / (k + 1),
                                bins,
                            )

                    for t in x_t:
                        del t
                    del y_pred

                    sample_dep[-1].append(
                        qd.sample_posterior(y_pred_dep, bins).cpu().numpy()[:, 0]
                    )
                    quantiles_dep[-1].append(
                        qd.posterior_quantiles(y_pred_dep, bins, quantiles)
                        .cpu()
                        .numpy()
                    )
                    mean_dep[-1].append(
                        qd.posterior_mean(y_pred_dep, bins).cpu().numpy()
                    )

                    sample_indep[-1].append(
                        qd.sample_posterior(y_pred_indep, bins).cpu().numpy()[:, 0]
                    )
                    quantiles_indep[-1].append(
                        qd.posterior_quantiles(y_pred_indep, bins, quantiles)
                        .cpu()
                        .numpy()
                    )
                    mean_indep[-1].append(
                        qd.posterior_mean(y_pred_indep, bins).cpu().numpy()
                    )

                    if self.correction:
                        y_pred_dep_c, y_pred_indep_c = self.correction(
                            y_pred_dep, y_pred_indep
                        )
                        y_pred_dep_c = qd.normalize(y_pred_dep_c, bins, 1, True)
                        sample_dep_c[-1].append(
                            qd.sample_posterior(y_pred_dep_c, bins).cpu().numpy()[:, 0]
                        )
                        quantiles_dep_c[-1].append(
                            qd.posterior_quantiles(y_pred_dep_c, bins, quantiles)
                            .cpu()
                            .numpy()
                        )
                        mean_dep_c[-1].append(
                            qd.posterior_mean(y_pred_dep_c, bins).cpu().numpy()
                        )

                        y_pred_indep_c = qd.normalize(y_pred_indep_c, bins, 1, True)
                        sample_indep_c[-1].append(
                            qd.sample_posterior(y_pred_indep_c, bins)
                            .cpu()
                            .numpy()[:, 0]
                        )
                        quantiles_indep_c[-1].append(
                            qd.posterior_quantiles(y_pred_indep_c, bins, quantiles)
                            .cpu()
                            .numpy()
                        )
                        mean_indep_c[-1].append(
                            qd.posterior_mean(y_pred_indep_c, bins).cpu().numpy()
                        )
                        del y_pred_dep_c
                        del y_pred_indep_c

                    del y_pred_dep
                    del y_pred_indep

        # Finally, concatenate over rows and columns.
        sample_dep = tiler.assemble(sample_dep)
        quantiles_dep = tiler.assemble(quantiles_dep).transpose([0, 2, 3, 1])
        mean_dep = tiler.assemble(mean_dep)

        sample_indep = tiler.assemble(sample_indep)
        quantiles_indep = tiler.assemble(quantiles_indep).transpose([0, 2, 3, 1])
        mean_indep = tiler.assemble(mean_indep)

        dims = ("time", "x", "y")
        dims_r = ("time", "x", "y")
        if mean_dep.shape[-2] != input_data.data.x.size:
            dims_r = ("time", "x_4", "y_4")

        results = xr.Dataset(
            {
                "time": ("time", input_data.data.time.mean("time").data.reshape((1,))),
                "n_inputs": (("time",), [n_inputs]),
                "latitude": input_data.data.latitude.mean("time"),
                "longitude": input_data.data.longitude.mean("time"),
                "quantiles": (("quantiles",), quantiles),
                "mean_dep": (dims_r, mean_dep),
                "sample_dep": (dims_r, sample_dep),
                "quantiles_dep": (dims_r + ("quantiles",), quantiles_dep),
                "mean_indep": (dims_r, mean_indep),
                "sample_indep": (dims_r, sample_indep),
                "quantiles_indep": (dims_r + ("quantiles",), quantiles_indep),
            }
        )

        if self.correction:
            # Finally, concatenate over rows and columns.
            sample_dep_c = tiler.assemble(sample_dep_c)
            quantiles_dep_c = tiler.assemble(quantiles_dep_c).transpose([0, 2, 3, 1])
            mean_dep_c = tiler.assemble(mean_dep_c)

            sample_indep_c = tiler.assemble(sample_indep_c)
            quantiles_indep_c = tiler.assemble(quantiles_indep_c).transpose([0, 2, 3, 1])
            mean_indep_c = tiler.assemble(mean_indep_c)

            results["mean_dep_c"] = (dims_r, mean_dep_c)
            results["sample_dep_c"] = (dims_r, sample_dep_c)
            results["quantiles_dep_c"] = (dims_r + ("quantiles",), quantiles_dep_c)
            results["mean_indep_c"] = (dims_r, mean_indep_c)
            results["sample_indep_c"] = (dims_r, sample_indep_c)
            results["quantiles_indep_c"] = (dims_r + ("quantiles",), quantiles_indep_c)

        results.attrs["any_invalid"] = any_invalid
        results.attrs["any_ir_invalid"] = any_ir_invalid

        return results

    def run(self):
        """
        Run retrieval and return results as 'xarray.Dataset'.
        """
        results = []
        for f in self.input_files:
            results.append(self._run_file(f))
        return xr.concat(results, "time")


######################################################################
# Model evaluator
######################################################################


class Evaluator:
    """
    Processor class to evaluate the Hydronn model on test data.
    """

    def __init__(self, input_files, model, normalizer, device="cuda", resolution=2, ir=False):
        """
        Args:
            input_files: The list of input files for which to run the
                retrieval.
            model: The model to evaluate.
            normalizer: The normalizer object to use to normalize the
                inputs.
            device: The device on which to run the retrieval.
            resolution: The resolution of the model.
            ir: Flag indicating whether the model is an IR-only model.
        """
        self.input_files = sorted(input_files)
        self.model = model
        self.normalizer = normalizer
        self.device = device
        self.resolution = resolution
        self.ir = ir

    def _run_file(self, input_file):
        """
        This function implements the evaluation for a single training data file.

        Args:
            input_file: Filename of the training data file.

        Returns:
            A 'xarray.Dataset' containing the results of the evaluation.
        """
        input_data = HydronnDataset(
            input_file,
            normalizer=self.normalizer,
            batch_size=32,
            resolution=self.resolution,
            shuffle=False,
            ir=self.ir,
            augment=False
        )
        tau = [
            0.01,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.99,
            0.999,
        ]
        device = self.device

        bins = torch.Tensor(self.model.bins).to(device)
        model = self.model.model.to(device)

        means = []
        quantiles = []
        samples = []
        truth = []

        for i in range(len(input_data)):
            with torch.no_grad():
                x, y = input_data[i]
                x = [t.to(device) for t in x]
                y_pdf = model(x)
                y_pdf = self.model._post_process_prediction(y_pdf, bins)

                # Posterior mean
                y_mean = qd.posterior_mean(y_pdf, bins=bins)
                means.append(y_mean.cpu().numpy())

                # Quantiles
                y_quants = qd.posterior_quantiles(y_pdf, quantiles=tau, bins=bins)
                quantiles.append(y_quants.cpu().numpy().transpose([0, 2, 3, 1]))

                # Samples
                y_samples = qd.sample_posterior(y_pdf, bins=bins)
                samples.append(y_samples.cpu().numpy()[:, 0])

                truth.append(y.cpu().numpy())

        means = np.concatenate(means)
        quantiles = np.concatenate(quantiles)
        samples = np.concatenate(samples)
        truth = np.concatenate(truth)

        dims = ("samples", "y", "x")
        results = xr.Dataset(
            {
                "surface_precip": (dims, means),
                "surface_precip_quantiles": (dims + ("quantiles",), quantiles),
                "surface_precip_samples": (dims, samples),
                "surface_precip_true": (dims, truth),
            }
        )

        # Copy some info from input data.
        results["time"] = (("samples",), input_data.time)
        results["x_coords"] = (("samples", "x"), input_data.x_coords)
        results["y_coords"] = (("samples", "y"), input_data.y_coords)

        return results

    def run(self):
        """
        Run retrieval and return results as 'xarray.Dataset'.
        """
        results = []
        for f in self.input_files:
            results.append(self._run_file(f))
        return xr.concat(results, "samples")


def retrieve(model,
             retrieval_input,
             quantiles=None,
             device="cpu",
             tile_size=None,
             overlap=0):
        """
        Run Hydronn retrieval a single GOES 16 observations.

        Args:
            retrieval_input: The normalized retrieval input.
            quantiles: A list of quantile fractions defining quantiles to
                include in the output.
            tile_size: Size of the tiles to use for the retrieval.
            overlap: Overlap to use for neighboring tiles.

        Return:
            An 'xarray.Dataset' containing the retrieval results.
        """
        taus = [
            0.01,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.99,
            0.999,
        ]

        bins = torch.Tensor(model.bins).to(device)
        d_bins = bins[1:] - bins[:-1]

        # If not tile size is given, run retrieval on full input.
        if tile_size is None:
            tile_size = (1920, 1920)

        tiler = Tiler(
            retrieval_input,
            tile_size=tile_size,
            overlap=overlap,
            resolution=model.resolution,
        )

        sample = []
        quantiles = []
        mean = []

        model.model.to(device)

        for i in range(tiler.M):

            sample.append([])
            quantiles.append([])
            mean.append([])

            for j in range(tiler.N):

                with torch.no_grad():
                    # Retrieve tile
                    x_t = tiler.get_tile(i, j)
                    x_t = [t.to(device) for t in x_t]

                    y_pred = model.model(x_t)[(...,)]
                    y_pred = model._post_process_prediction(y_pred, bins)

                    for t in x_t:
                        del t

                    sample[-1].append(
                        qd.sample_posterior(y_pred, bins).cpu().numpy()[:, 0]
                    )
                    quantiles[-1].append(
                        qd.posterior_quantiles(y_pred, bins, taus)
                        .cpu()
                        .numpy()
                    )
                    mean[-1].append(
                        qd.posterior_mean(y_pred, bins).cpu().numpy()
                    )

                    del y_pred

        # Finally, concatenate over rows and columns.
        sample = tiler.assemble(sample)
        quantiles = tiler.assemble(quantiles).transpose([0, 2, 3, 1])
        mean = tiler.assemble(mean)

        dims = ("time", "x", "y")
        dims_r = ("time", "x", "y")
        if mean.shape[-2] != retrieval_input[0].shape[-2]:
            dims_r = ("time", "x_4", "y_4")

        results = xr.Dataset(
            {
                "tau": (("tau",), taus),
                "surface_precip": (dims_r, mean),
                "surface_precip_sampled": (dims_r, sample),
                "surface_precip_quantiles": (dims_r + ("tau",), quantiles),
            }
        )

        return results
