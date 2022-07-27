"""
====================
hydronn.bin.retrieve
====================

This module implements the command line application to run the hydronn
retrieval.
"""
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor
from datetime import (datetime, timedelta)
import logging
import multiprocessing as mp
from multiprocessing import Queue, Manager, Process, Lock
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory, mkdtemp
import gc

from rich.progress import track
import torch
import xarray as xr


LOGGER = logging.getLogger(__name__)


FILE_PATTERN = re.compile(r".*\.nc(.gz)?")


def add_parser(subparsers):
    """
    Add parser for 'retrieve' command to top-level CLI.
    """
    parser = subparsers.add_parser(
        'retrieve',
        help="Run the hydronn retrieval",
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
    parser.add_argument(
        '--correction', metavar='path', type=str,
        help='Optional path to correction file to apply.',
        default=None
    )

    parser.set_defaults(func=run)

def process_file(
        model,
        input_file,
        output_file,
        tile_size,
        overlap,
        device_queue,
        correction):
    """
    Process an input file containing one hour of input observations.

    Args:
        model: Path to the Hydronn model to use for the processing.
        input_file: The file to process.
        output_file: The file to which to write the results.
        tile_size: The tile size to use for processing.
        overlap: The overlap to use for the tiling.
        device_queue: Queue with available devices.
        correction: Path to the correction file to apply.
    """
    from hydronn.retrieval import Retrieval, InputFile
    from hydronn.utils import save_and_compress
    from quantnn.qrnn import QRNN
    import torch
    model = QRNN.load(model)
    normalizer = model.normalizer
    retrieval = Retrieval([InputFile(input_file, normalizer)],
                            model,
                            normalizer,
                            tile_size=tile_size,
                            overlap=overlap,
                            device="cpu",
                            correction=correction)

    print("Waiting for device")
    device = device_queue.get()
    try:
        LOGGER.info(
            "Starting processing of file '%s' on device '%s'.",
            input_file,
            device
        )
        retrieval.device = device
        results = retrieval.run()
        del retrieval
        model.model.cpu()
        gc.collect()
    finally:
        device_queue.put(device)
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    if str(output_file).endswith(".gz"):
        output_file = str(output_file)[:-3]
    save_and_compress(results, output_file)


def run(args):
    """
    This function implements the actual execution of retrieval for
    the given input.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from quantnn.qrnn import QRNN
    from hydronn.retrieval import Retrieval, InputFile
    from hydronn.utils import save_and_compress

    mp.set_start_method("spawn", force=True)

    model = Path(args.model)
    if not model.exists():
        LOGGER.error("The model %s' does not exist.", model)

    input_path = Path(args.input_path)
    if not input_path.exists():
        LOGGER.error("The input path '%s' does not exist.", input_path)

    output_path = Path(args.output_path)
    if not output_path.exists():
        LOGGER.error("The output path '%s' does not exist.", output_path)

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
    correction = args.correction
    if correction is not None and not Path(correction).exists():
        LOGGER.error(
            "The provided correction file '%s' doesn't exist.",
            correction
        )
        return 1

    manager = mp.Manager()
    queue = manager.Queue()
    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            queue.put(f"cuda:{i}")
    else:
        for i in range(4):
            queue.put("cpu")

    available_gpus = [
        torch.cuda.device(i) for i in range(torch.cuda.device_count())
    ]
    pool = ProcessPoolExecutor(
        max_workers=4
    )

    tasks = []
    for input_file, output_file in zip(input_files, output_files):
        output_compressed = Path(str(output_file) + ".gz")
        if not (output_file.exists() or output_compressed.exists()):
            tasks.append(pool.submit(
                process_file,
                model,
                input_file, output_file, tile_size, overlap, queue,
                correction
            ))
        else:
            print(f"File '{output_file}' already exists. Skipping.")

    for task in tasks:
        task.result()


