"""
====================
hydronn.bin.evaluate
====================

This module implements the command line application to evaluate a
hydronn model.
"""
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, ThreadPoolExecutor
from datetime import (datetime, timedelta)
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory, mkdtemp

from rich.progress import track
import xarray as xr


LOGGER = logging.getLogger(__name__)


FILE_PATTERN = re.compile(r".*\.nc(.gz)?")


def add_parser(subparsers):
    """
    Add parser for 'retrieve' command to top-level CLI.
    """
    parser = subparsers.add_parser(
            'evaluate',
            description='Evaluate model on test data.'
            )
    parser.add_argument(
        'model', metavar='model', type=str,
        help='Path to the model to use for the retrieval.'
    )
    parser.add_argument(
        'input_path', metavar='input_path', type=str,
        help='Folder containing the retrieval input data.'
    )
    parser.add_argument(
        'output_path', metavar='output_path', type=str,
        help='Where to store the retrieval output.'
    )
    parser.add_argument(
        '--device', metavar='device', type=str,
        help='Device on which to run the retrieval.',
        default="cuda"
    )
    parser.add_argument(
        "--ir",
        action="store_true",
        help="Whether the evaluation is of an IR only model."
    )
    parser.set_defaults(func=run)


def run(args):
    """
    This function implements the actual execution of retrieval for
    the given input.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from quantnn.qrnn import QRNN
    from hydronn.retrieval import Evaluator
    from hydronn.utils import save_and_compress

    model = Path(args.model)
    if not model.exists():
        LOGGER.error("The model %s' does not exist.", model)
        return 1

    resolution = 4 if "4" in str(model.name) else 2

    model = QRNN.load(model)
    normalizer = model.normalizer


    input_path = Path(args.input_path)
    if not input_path.exists():
        LOGGER.error("The input path '%s' does not exist.", input_path)
        return 1

    output_path = Path(args.output_path)
    if output_path.exists() and output_path.is_dir():
        LOGGER.error("The output path must not be a directory", output_path)
        return 1

    parent = output_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    device = args.device

    input_files = []
    if input_path.is_file():
        input_files.append(Path(input_path))
    else:
        for f in input_path.glob("**/*"):
            if FILE_PATTERN.match(f.name):
                input_files.append(f)

    evaluator = Evaluator(input_files,
                          model,
                          normalizer,
                          device=device,
                          resolution=resolution,
                          ir=args.ir)
    results = evaluator.run()
    save_and_compress(results, output_path)
