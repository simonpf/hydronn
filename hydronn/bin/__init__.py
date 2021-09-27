"""
===========
hydronn.bin
===========

This module implements the 'hydronn' command line applitcation.
"""
import argparse
import sys

def hydronn():
    """
    This function implements the top-level command line interface for the
    'gprof_nn' package. It serves as the global entry point to execute
    any of the available sub-commands.
    """
    from hydronn.bin import extract_data
    from hydronn.bin import train

    description = ("HYDRONN: A NRT precipitation retrieval for Brazil.")
    parser = argparse.ArgumentParser(prog='hydronn', description=description)

    subparsers = parser.add_subparsers(help='Sub-commands')

    extract_data.add_parser(subparsers)
    train.add_parser(subparsers)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()
    args.func(args)

