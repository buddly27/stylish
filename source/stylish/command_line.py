# :coding: utf-8

import sys
import argparse
import logging

import stylish.vgg
import stylish.train
import stylish.filesystem


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

    subparsers = parser.add_subparsers(
        title="Commands",
        dest="subcommands"
    )

    train_parser = subparsers.add_parser(
        "train", help="Train a style generator model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    train_parser.add_argument(
        "--vgg19-path",
        help="Path to Vgg19 pre-trained model in the MatConvNet data format.",
        metavar="PATH",
        required=True
    )

    train_parser.add_argument(
        "--style-target",
        help="Path to image from which the style features will be extracted.",
        metavar="PATH",
        required=True
    )

    train_parser.add_argument(
        "--content-target",
        help=(
            "Path to a folder containing images from which the content "
            "features will be extracted."
        ),
        metavar="PATH",
        nargs="+",
        required=True
    )

    train_parser.add_argument(
        "--output-model",
        help="Path to save the output trained model.",
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

    if namespace.subcommands == "train":

        layers, mean_pixel = stylish.vgg.extract_data(namespace.vgg19_path)

        # Extract targeted images for training.
        content_targets = stylish.filesystem.fetch_images(
            namespace.content_target
        )

        # Ensure that the output model path exist and is accessible.
        stylish.filesystem.ensure_directory_access(namespace.output_model)

        # Train the model and export it.
        model_path = stylish.train.extract_model(
            namespace.style_target, content_targets, layers, mean_pixel,
            namespace.output_model
        )

        logging.info("Model trained: {}".format(model_path))
