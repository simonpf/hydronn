"""
=================
hydronn.bin.train
=================

This sub-module implements the 'train' sub-command of the
'hydronn' command line application.
"""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'train' command to top-level parser. This function
    is called from the top-level parser defined in 'hydronn.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "train",
        help="Train a hydronn retrieval model.",
        description=(
            """
            Train the hydronn retrieval model.
            """
            )
    )

    parser.add_argument(
        'training_data',
        metavar='training_data',
        type=str,
        help='Path to training data.'
    )
    parser.add_argument(
        'validation_data',
        metavar='validation_data',
        type=str,
        help='Path to validation data.'
    )
    parser.add_argument(
        'output',
        metavar='output',
        type=str,
        nargs=1,
        help='Where to store the model.'
    )

    parser.add_argument(
        "--resolution",
        metavar='n',
        type=int,
        default=2,
        help=("The resolution in km at which to perform the retrieval.")
    )
    parser.add_argument(
        "--n_features_body",
        metavar='n',
        type=int,
        default=256,
        help=("The number of features in the body of the network.")
    )
    parser.add_argument(
        '--n_layers_head',
        metavar='n',
        type=int,
        default=4,
        help='The number of layers in the head of the network.'
    )
    parser.add_argument(
        '--n_features_head',
        metavar='n',
        type=int,
        default=128,
        help="The number of features in the head of the network."
    )
    parser.add_argument(
        '--n_blocks',
        metavar='N',
        type=int,
        nargs="*",
        default=[2],
        help="The number of block in the stages of the encoder."
    )
    parser.add_argument(
        '--learning_rate',
        metavar='lr',
        type=float,
        nargs="*",
        default=[0.0005, 0.0005, 0.0001],
        help='The learning rates to use during training.'
    )
    parser.add_argument(
        '--n_epochs',
        metavar='lr',
        type=int,
        nargs="*",
        default=[10, 10, 10],
        help='The learning rates to use during training.'
    )
    parser.add_argument(
        '--no_lr_schedule',
        action="store_true",
        help='Disable learning rate schedule.'
    )
    parser.add_argument(
        "--ir",
        action="store_true",
        help=("Whether to train an IR only retrieval.")
    )

    # Other
    parser.add_argument(
        '--device', metavar="device", type=str, nargs=1,
        help="The name of the device on which to run the training",
        default="cpu"
    )
    parser.add_argument(
        '--batch_size', metavar="n", type=int, nargs=1,
        help="The batch size to use for training."
    )
    parser.set_defaults(func=run)


def run(args):
    """
    Run the training.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    import torch
    from quantnn.qrnn import QRNN
    from quantnn.drnn import DRNN
    from quantnn.normalizer import Normalizer
    from quantnn.data import DataFolder
    from quantnn.transformations import LogLinear
    from quantnn.models.pytorch.logging import TensorBoardLogger
    from quantnn.metrics import ScatterPlot
    from hydronn.models import Hydronn2, Hydronn4, Hydronn4IR
    from hydronn.data.training_data import HydronnDataset
    from hydronn.definitions import HYDRONN_DATA_PATH
    from torch import optim

    training_data = args.training_data[0]
    validation_data = args.validation_data[0]

    #
    # Configuration
    #

    # Check output path and define model name if necessary.
    output = Path(args.output[0])

    if output.is_dir() and not output.exists():
        LOGGER.error(
            "The output path '%s' doesn't exist.", output
        )
        return 1

    if not output.is_dir() and not output.parent.exists():
        LOGGER.error(
            "The output path '%s' doesn't exist.", output.parent
        )
        return 1

    resolution = args.resolution
    n_blocks = args.n_blocks[0]
    n_features_body = args.n_features_body
    n_layers_head = args.n_layers_head
    n_features_head = args.n_features_head
    ir = args.ir

    LOGGER.info("Training %s km model.", resolution)
    if ir:
        LOGGER.info("Training IR configuration.")

    if output.is_dir():
        network_name = (
            f"hydronn_{n_blocks}_{n_features_body}_{n_layers_head}"
            f"_{n_features_head}.pckl"
        )
        output = output / network_name

    if isinstance(n_blocks, list):
        if len(n_blocks) < 7:
            n_blocks = n_blocks * 7

    training_data = args.training_data
    validation_data = args.validation_data

    device = args.device[0]
    batch_size = args.batch_size[0]

    n_epochs = args.n_epochs
    lr = args.learning_rate
    no_schedule = args.no_lr_schedule

    if len(n_epochs) == 1:
        n_epochs = n_epochs * len(lr)
    if len(lr) == 1:
        lr = lr * len(n_epochs)

    #
    # Load training data.
    #

    dataset_factory = HydronnDataset
    normalizer_path = (HYDRONN_DATA_PATH/ "normalizer.pckl")
    normalizer = Normalizer.load(normalizer_path)
    kwargs = {
        "batch_size": batch_size,
        "normalizer": normalizer,
        "augment": True,
        "resolution": resolution,
        "ir": ir
    }
    training_data = DataFolder(
        training_data,
        dataset_factory,
        queue_size=32,
        kwargs=kwargs,
        n_workers=2)

    kwargs = {
        "batch_size": 4 * batch_size,
        "normalizer": normalizer,
        "augment": False,
        "resolution": resolution,
        "ir": ir
    }
    validation_data = DataFolder(
        validation_data,
        dataset_factory,
        queue_size=32,
        kwargs=kwargs,
        n_workers=1
    )

    ###############################################################################
    # Prepare in- and output.
    ###############################################################################

    #
    # Create neural network model
    #

    if Path(output).exists():
        try:
            xrnn = QRNN.load(output)
            LOGGER.info(
                f"Continuing training of existing model {output}."
            )
        except Exception:
            xrnn = None
    else:
        xrnn = None

    if xrnn is None:
        bins = np.logspace(-3, 3, 129)
        if resolution == 2:
            model = Hydronn2(
                128, n_blocks, n_features_body,
                n_layers_head, n_features_head
            )
        else:
            if ir:
                model = Hydronn4IR(
                    128, n_blocks, n_features_body,
                    n_layers_head, n_features_head
                )
            else:
                model = Hydronn4(
                    128, n_blocks, n_features_body,
                    n_layers_head, n_features_head
                )

        xrnn = DRNN(model=model, bins=bins)

    model = xrnn.model
    model.normalizer = normalizer

    ###############################################################################
    # Run the training.
    ###############################################################################

    n_epochs_tot = sum(n_epochs)
    logger = TensorBoardLogger(n_epochs_tot, log_rate=1)
    logger.set_attributes({
        "n_blocks": n_blocks,
        "n_neurons_body": n_features_body,
        "n_layers_head": n_layers_head,
        "n_neurons_head": n_features_head,
        "optimizer": "adam"
        })

    metrics = ["MeanSquaredError", "Bias", "CalibrationPlot", "CRPS"]
    scatter_plot = ScatterPlot(log_scale=True)
    metrics.append(scatter_plot)

    for n, r in zip(n_epochs, lr):
        LOGGER.info(
            f"Starting training for {n} epochs with learning rate {r}"
        )
        optimizer = optim.Adam(model.parameters(), lr=r)
        if no_schedule:
            scheduler = None
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n)
        xrnn.train(training_data=training_data,
                   validation_data=validation_data,
                   n_epochs=n,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   logger=logger,
                   metrics=metrics,
                   device=device,
                   mask=-1)
        LOGGER.info(
            f"Saving training network to {output}."
        )
        xrnn.normalizer = normalizer
        xrnn.resolution = resolution
        xrnn.save(output)
