"""
hydronn.retrieval
=================

Functionality for running the hydronn retrieval.
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


class InputFile:
    """
    Interface class to load the retrieval input data from the NetCDF files
    used to store the observations.
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
        self.data = xr.load_dataset(filename).sortby("time")
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
        t = t_start - t_end

        low_res = []
        for c in LOW_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in self.data:
                x = self.data[channel_name].data[t_start:t_end]
            else:
                x = np.zeros((t, m, n), dytpe=np.float32)
            low_res.append(x)
        low_res = np.stack(low_res, axis=1)

        med_res = []
        for c in MED_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in self.data:
                x = self.data[channel_name].data[t_start:t_end]
            else:
                x = np.zeros((t, 2 * m, 2 * n), dytpe=np.float32)
            med_res.append(x)
        med_res = np.stack(med_res, axis=1)

        hi_res = []
        for c in HI_RES_CHANNELS:
            channel_name = f"C{c:02}"
            if channel_name in self.data:
                x = self.data[channel_name].data[t_start:t_end]
            else:
                x = np.zeros((t, 4 * m, 4 * n), dytpe=np.float32)
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
        if self.batch_size < 0:
            return self.get_input_data(0, self.data.time.size)
        else:
            t_start = i * self.batch_size
            t_end = t_start + self.batch_size
            return self.get_input_data(t_start, t_end)


class Retrieval:
    """
    Processor class to run the retrieval for a list of input files.
    """
    def __init__(self,
                 input_files,
                 model,
                 normalizer):
        """
        Args:
            input_files: The list of input files for which to run the
                retrieval.
            model: The model to use for the retrieval.
            normalizer: The normalizer object to use to normalize the
                inputs.
        """
        self.input_files = sorted(input_files)
        self.model = model
        self.normalizer = normalizer

    def _run_file(self, input_file):
        """
        Run retrieval for a single input files.
        """
        input_data = InputFile(input_file,
                               self.normalizer,
                               1)

        y_pred_dep = None
        y_mean_dep = None
        y_sample_dep = None
        y_pred_indep = None
        y_mean_indep = None
        y_sample_indep = None

        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

        n = len(input_data)
        bins = torch.Tensor(self.model.bins)
        bins_acc = (1 / n) * bins

        for i in range(len(input_data)):
            x = input_data[i]
            with torch.no_grad():

                y_pred = self.model.model(x)
                y_pred = self.model._post_process_prediction(
                    y_pred,
                    bins
                )

                if y_pred_dep is None:
                    y_pred_dep = (1 / n) * y_pred
                    y_pred_indep = y_pred
                else:
                    y_pred_dep = y_pred_dep + (1 / n) * y_pred
                    y_pred_indep = qd.add(
                        y_pred_indep, bins, y_pred, bins, bins
                    )

        sample_dep = qd.sample_posterior(
            y_pred_dep, bins
        ).cpu().numpy()[:, 0]
        quantiles_dep = qd.posterior_quantiles(
            y_pred_dep, bins, quantiles
        ).cpu().numpy().transpose([0, 2, 3, 1])
        mean_dep = qd.posterior_mean(
            y_pred_dep, bins
        ).cpu().numpy()

        y_pred_indep = n * y_pred_indep
        sample_indep = qd.sample_posterior(
            y_pred_indep, bins_acc
        ).cpu().numpy()[:, 0]
        quantiles_indep = qd.posterior_quantiles(
            y_pred_indep, bins_acc, quantiles
        ).cpu().numpy().transpose([0, 2, 3, 1])
        mean_indep = qd.posterior_mean(
            y_pred_indep, bins_acc
        ).cpu().numpy()

        dims = ("time", "x", "y")
        results = xr.Dataset({
            "time":  ("time", input_data.data.time.mean("time").data.reshape((1,))),
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

        return results

    def run(self):
        """
        Run retrieval and return results as 'xarray.Dataset'.
        """
        results = []
        for f in self.input_files:
            results.append(self._run_file(f))
        return xr.concat(results, "time")
