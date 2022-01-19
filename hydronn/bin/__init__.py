"""
===========
hydronn.bin
===========

This module implements the 'hydronn' command line applitcation.
"""
import argparse
import logging
import sys

def hydronn():
    """
    This function implements the top-level command line interface for the
    'gprof_nn' package. It serves as the global entry point to execute
    any of the available sub-commands.
    """
    from hydronn.bin import extract_data
    from hydronn.bin import extract_retrieval_data
    from hydronn.bin import train
    from hydronn.bin import retrieve
    from hydronn.bin import evaluate

    description = ("HYDRONN: A NRT precipitation retrieval for Brazil.")
    parser = argparse.ArgumentParser(prog='hydronn', description=description)

    subparsers = parser.add_subparsers(help='Sub-commands')

    extract_data.add_parser(subparsers)
    extract_retrieval_data.add_parser(subparsers)
    train.add_parser(subparsers)
    retrieve.add_parser(subparsers)
    evaluate.add_parser(subparsers)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    logging.basicConfig(level="INFO")
    args = parser.parse_args()
    args.func(args)

