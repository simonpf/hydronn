"""
====================
hydronn.bin.retrieve
====================

This module implements the command line application to run the hydronn
retrieval.
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
            'retrieve',
            description='Run the hydronn retrieval.'
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
        '--tile_size', metavar='N', type=int,
        help='Size of the tiles that are processed simultaneously',
        nargs="+",
        default=[256],
    )
    parser.add_argument(
        '--overlap', metavar='N', type=int,
        help='Overlap between neighboring tiles',
        default=32
    )
    parser.add_argument(
        '--subset', metavar='even / odd', type=str,
        help='Retrieve only even or odd days.'
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
    from hydronn.retrieval import Retrieval
    from hydronn.utils import save_and_compress

    model = Path(args.model)
    if not model.exists():
        LOGGER.error("The model %s' does not exist.", model)
    model = QRNN.load(model)
    normalizer = model.normalizer

    input_path = Path(args.input_path)
    if not input_path.exists():
        LOGGER.error("The input path '%s' does not exist.", input_path)

    output_path = Path(args.output_path)
    if not output_path.exists():
        LOGGER.error("The output path '%s' does not exist.", input_path)

    input_files = []
    output_files = []
    if input_path.is_file():
        input_files.append(Path(input_path))
    else:
        for f in input_path.glob("**/*"):
            if FILE_PATTERN.match(f.name):
                input_files.append(f)

    subset = args.subset
    if subset is not None:
        if subset.lower() == "even":
            m = 0
        else:
            m = 1

        input_files_filtered = []
        for f in input_files:
            day = int(f.name.split("_")[4])
            if day % 2 == m:
                input_files_filtered.append(f)
        input_files = input_files_filtered

    for f in input_files:
        output_parent = output_path / f.relative_to(input_path).parent
        output_file = f.name.replace("input", "output")
        if output_file.endswith(".gz"):
            output_file = output_file[:-3]
        output_files.append(output_parent / output_file)

    tile_size = args.tile_size
    overlap = args.overlap
    device = args.device

    for f, o in zip(input_files, output_files):
        print(f, o)
        if not o.exists():
            print(f"File '{f}' doesn't exist. continuing.")
            retrieval = Retrieval([f],
                                  model,
                                  normalizer,
                                  tile_size=tile_size,
                                  overlap=overlap,
                                  device=device)
            results = retrieval.run()
            if not o.parent.exists():
                o.parent.mkdir(parents=True)
            if str(o).endswith(".gz"):
                o = str(o)[:-3]
            save_and_compress(results, o)
            print(f"Finished processing input file '{f}'")
        else:
            print(f"File '{f}' already exists. Skipping.")

