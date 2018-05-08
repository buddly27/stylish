# :coding: utf-8

import sys
import argparse
import logging

import stylish.vgg
import stylish.filesystem
import stylish.train


def construct_parser():
    """Return argument parser."""
    parser = argparse.ArgumentParser(
        prog="stylish",
        description="Style transfer using deep neural network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-v", "--verbosity",
        help="Set the logging output verbosity.",
        choices=["debug", "info", "warning", "error"],
        default="info"
    )

    parser.add_argument(
        "--vgg19-path",
        help="Path to Vgg19 pre-trained model in the MatConvNet data format.",
        metavar="PATH",
        required=True
    )

    parser.add_argument(
        "--style-image-path",
        help="Path to image from which the style will be extracted.",
        metavar="PATH",
        required=True
    )

    return parser


def _log_level(name):
    """Return log level from *name*.
    """
    if name == "debug":
        return logging.DEBUG

    elif name == "info":
        return logging.INFO

    elif name == "warning":
        return logging.WARNING

    elif name == "error":
        return logging.ERROR

    else:
        raise ValueError(
            "The logging level is incorrect: {!r}".format(name)
        )


def main(arguments=None):
    """Stylish command line interface."""
    if arguments is None:
        arguments = []

    # Process arguments.
    parser = construct_parser()
    namespace = parser.parse_args(arguments)

    # Initiate logger.
    logging.basicConfig(
        stream=sys.stderr, level=_log_level(namespace.verbosity),
        format="%(levelname)s: %(message)s"
    )

    image_matrix = stylish.filesystem.load_image(namespace.style_image_path)
    layers, mean_pixel = stylish.vgg.extract_data(namespace.vgg19_path)

    stylish.train.execute(image_matrix, layers, mean_pixel)
