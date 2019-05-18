# :coding: utf-8


import click

import stylish.logging
import stylish.vgg
import stylish.train
import stylish.feed_forward
import stylish.filesystem
from stylish import __version__


#: Click default context for all commands.
CONTEXT_SETTINGS = dict(
    max_content_width=90,
    help_option_names=["-h", "--help"],
)


@click.group(
    context_settings=CONTEXT_SETTINGS,
    help="Style transfer using deep neural network",
)
@click.version_option(version=__version__)
@click.option(
    "-v", "--verbosity",
    help="Set the logging output verbosity.",
    type=click.Choice(stylish.logging.levels),
    default="info",
    show_default=True
)
def main(**kwargs):
    """Main entry point for the command line interface."""
    stylish.logging.configure()

    # Set verbosity level.
    stylish.logging.root.handlers["stderr"].filterer.min = kwargs["verbosity"]


@main.command(
    name="train",
    help="Train a style generator model.",
    context_settings=CONTEXT_SETTINGS
)
@click.option(
    "--vgg",
    help="Path to Vgg19 pre-trained model in the MatConvNet data format.",
    metavar="PATH",
    type=click.Path(),
    required=True
)
@click.option(
    "-s", "--style",
    help="Path to image from which the style features will be extracted.",
    metavar="PATH",
    type=click.Path(),
    required=True
)
@click.option(
    "-c", "--content",
    help=(
        "Path to a folder containing images from which the content "
        "features will be extracted."
    ),
    metavar="PATH",
    type=click.Path(),
    required=True
)
@click.option(
    "-o", "--output",
    help="Path to folder in which the trained model will be saved.",
    metavar="PATH",
    type=click.Path(),
    required=True
)
def stylish_train(**kwargs):
    """Train a style generator model."""
    logger = stylish.logging.Logger(__name__ + ".stylish_train")

    layers, mean_pixel = stylish.vgg.extract_data(kwargs["vgg"])

    # Extract targeted images for training.
    content_targets = stylish.filesystem.fetch_images(kwargs["content"])

    # Ensure that the output path exist and is accessible.
    stylish.filesystem.ensure_directory_access(kwargs["output"])

    # Train the model and export it.
    model_path = stylish.train.extract_model(
        kwargs["style"], content_targets, layers, mean_pixel, kwargs["output"]
    )

    logger.info("Model trained: {}".format(model_path))


@main.command(
    name="apply",
    help="Apply a style generator model to an image.",
    context_settings=CONTEXT_SETTINGS
)
@click.option(
    "--model",
    help=(
        "Path to trained style generator model which will be used to "
        "apply the style."
    ),
    metavar="PATH",
    type=click.Path(),
    required=True
)
@click.option(
    "-i", "--input",
    help="Path to image to transform.",
    metavar="PATH",
    type=click.Path(),
    required=True
)
@click.option(
    "-o", "--output",
    help="Path to folder in which the transformed image will be saved.",
    metavar="PATH",
    type=click.Path(),
    required=True
)
def stylish_apply(**kwargs):
    """Apply a style generator model to an image."""
    logger = stylish.logging.Logger(__name__ + ".stylish_apply")

    # Ensure that the output path exist and is accessible.
    stylish.filesystem.ensure_directory_access(kwargs["output"])

    stylish.feed_forward.transform_image(
        kwargs["model"], kwargs["input"], kwargs["output"]
    )

    logger.info("Model applied: {}".format(kwargs["model"]))
