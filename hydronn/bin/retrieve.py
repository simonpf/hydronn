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
from multiprocessing import Queue, Manager, Process, Lock
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory, mkdtemp
import gc

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
    parser.add_argument(
        '--correction', metavar='path', type=str,
        help='Optional path to correction file to apply.',
        default=None
    )

    parser.set_defaults(func=run)


def load_file(queue, input_file, normalizer, output_file):
    """
    Helper function for parallel loading of input data.

    Args:
        queue: Queue to put input data on.
        input_file: Name of the retrieval input file.
        normalizer: Normalizer object to use to normalize the input data.
        output_file: File to write the output to.
    """
    from hydronn.retrieval import InputFile
    queue.put((
        InputFile(input_file, normalizer, batch_size=6),
        output_file
    ))


def loader(task_queue, input_queue):
    from hydronn.retrieval import InputFile
    while task_queue.qsize():
        input_file, normalizer, output_file = task_queue.get()
        print(f"Loading {input_file}.")
        input_queue.put((
            InputFile(input_file, normalizer, batch_size=6),
            output_file
        ))
        print(f"Loaded {input_file}.")


_LOCK = None
def pool_init(lock):
    global _LOCK
    _LOCK = lock

def process_file(
        model,
        input_file,
        output_file,
        tile_size,
        overlap,
        device,
        correction):
        from hydronn.retrieval import Retrieval, InputFile
        from hydronn.utils import save_and_compress
        from quantnn.qrnn import QRNN
        import torch
        model = QRNN.load(model)
        normalizer = model.normalizer
        retrieval = Retrieval([input_file],
                              model,
                              normalizer,
                              tile_size=tile_size,
                              overlap=overlap,
                              device=device,
                              correction=correction)
        _LOCK.acquire()
        results = retrieval.run()
        del retrieval
        model.model.cpu()
        gc.collect()
        _LOCK.release()
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

    model = Path(args.model)
    if not model.exists():
        LOGGER.error("The model %s' does not exist.", model)

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
    correction = args.correction
    if correction is not None and not Path(correction).exists():
        LOGGER.error(
            "The provided correction file '%s' doesn't exist.",
            correction
        )
        return 1

    lock = Lock()
    pool = ProcessPoolExecutor(
        max_workers=4, initializer=pool_init, initargs=(lock,)
    )

    tasks = []
    for input_file, output_file in zip(input_files, output_files):
        output_compressed = Path(str(output_file) + ".gz")
        if not (output_file.exists() or output_compressed.exists()):
            tasks.append(pool.submit(
                process_file,
                model,
                input_file, output_file, tile_size, overlap, device,
                correction
            ))
        else:
            print(f"File '{output_file}' already exists. Skipping.")

    for task in tasks:
        task.result()

    ## Go through input an process on device.
    #for i in range(n_files):
    #    print(f"Running retrieval {i}.")
    #    retrieval = Retrieval([input_file],
    #                          model,
    #                          normalizer,
    #                          tile_size=tile_size,
    #                          overlap=overlap,
    #                          device=device,
    #                          correction=correction)
    #    results = retrieval.run()

    #    if not output_file.parent.exists():
    #        output_file.parent.mkdir(parents=True)
    #    if str(output_file).endswith(".gz"):
    #        output_file = str(output_file)[:-3]
    #    compress_pool.submit(save_and_compress, results, output_file)
    #    print(f"Finished processing file '{output_file}'")

    #compress_pool.shutdown(wait=True)


