# :coding: utf-8

import argparse

import mlog


def construct_parser():
    """Return argument parser."""
    parser = argparse.ArgumentParser(
        prog="stylish",
        description="Style transfer using deep neural network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Allow setting of logging level from arguments.
    parser.add_argument(
        "-v", "--verbosity",
        help="Set the logging output verbosity.",
        choices=mlog.levels,
        default="info"
    )

    return parser


def main(arguments=None):
    """Stylish command line interface."""
    if arguments is None:
        arguments = []

    mlog.configure()
    logger = mlog.Logger(__name__ + ".main")

    # Process arguments.
    parser = construct_parser()
    namespace = parser.parse_args(arguments)

    mlog.root.handlers["stderr"].filterer.filterers[0].min = (
        namespace.verbosity
    )

    logger.info("Hello from Stylish!")
