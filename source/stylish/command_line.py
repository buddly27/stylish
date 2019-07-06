# :coding: utf-8

import os
import shutil
import contextlib
import tempfile

import click
import requests

import stylish
import stylish.logging
import stylish.vgg
import stylish.filesystem
from stylish import __version__


#: Click default context for all commands.
CONTEXT_SETTINGS = dict(
    max_content_width=90,
    help_option_names=["-h", "--help"],
)


def _validate_train(_, option, value):
    """Custom validation for training options."""
    if option.name == "style":
        if not value or not os.path.isfile(value):
            raise click.BadParameter(
                "You need to indicate an image path to extract the style from."
            )

    elif option.name == "training":
        if not value or not os.path.isdir(value):
            raise click.BadParameter(
                "You need a training dataset to extract the content from in "
                "order to train a style generator model.\n"
                "\n"
                "You can use the following command to download the COCO "
                "training dataset:\n"
                "   >>> stylish download coco2014\n"
            )

    elif option.name == "vgg":
        if not value or not os.path.isfile(value):
            raise click.BadParameter(
                "You need a pre-trained Vgg19 model in the MatConvNet data "
                "format in order to train a style generator model.\n"
                "\n"
                "Please run the following command to download it:\n"
                "   >>> stylish download vgg19\n"
            )

    return value


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


@main.group(
    name="download",
    help=(
        """
        Download necessary elements to train a style generator model
        
        
        Example:
        
            \b
            stylish download vgg19
            stylish download coco2014 -o /tmp
            
        """
    ),
    short_help="Download necessary elements to train a model.",
    chain=True,
    context_settings=CONTEXT_SETTINGS
)
def stylish_download():
    """Download necessary elements to train a style generator model."""
    pass


@stylish_download.command(
    name="vgg19",
    help="Download pre-trained Vgg19 model (549MB).",
    context_settings=CONTEXT_SETTINGS
)
@click.option(
    "-o", "--output",
    help=(
        "Output path to save the element (Current directory is used by default)"
    ),
    metavar="PATH",
    type=click.Path(),
)
def stylish_download_vgg19(**kwargs):
    """Download pre-trained Vgg19 model."""
    logger = stylish.logging.Logger(__name__ + ".stylish_download_vgg19")

    # Pre-trained model source.
    name = "imagenet-vgg-verydeep-19.mat"
    uri = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
    path = os.path.join(kwargs.get("output") or os.getcwd(), name)

    logger.info(
        "Download pre-trained Vgg19 model:\n"
        "    source: {}\n"
        "    target: {}\n".format(uri, path)
    )

    # Download file.
    _download(logger, uri, path)


@stylish_download.command(
    name="coco2014",
    help="Download COCO 2014 Training dataset (13GB).",
    context_settings=CONTEXT_SETTINGS
)
@click.option(
    "-o", "--output",
    help=(
        "Output path to save the element (Current directory is used by default)"
    ),
    metavar="PATH",
    type=click.Path(),
)
def stylish_download_coco2014(**kwargs):
    """Download COCO 2014 Training dataset."""
    logger = stylish.logging.Logger(__name__ + ".stylish_download_coco2014")

    # Pre-trained model source.
    name = "train2014.zip"
    uri = "http://images.cocodataset.org/zips/train2014.zip"
    path = os.path.join(kwargs.get("output") or os.getcwd(), name)

    logger.info(
        "Download COCO 2014 Training dataset:\n"
        "    source: {}\n"
        "    target: {}\n".format(uri, path)
    )

    # Download file.
    _download(logger, uri, path)


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
    callback=_validate_train
)
@click.option(
    "-s", "--style",
    help="Path to image from which the style features will be extracted.",
    metavar="PATH",
    type=click.Path(),
    callback=_validate_train
)
@click.option(
    "-t", "--training",
    help="Path to a training dataset folder (e.g. COCO 2014).",
    metavar="PATH",
    type=click.Path(),
    callback=_validate_train
)
@click.option(
    "--limit",
    help="Maximum number of files to use from the training dataset folder.",
    type=int
)
@click.option(
    "-l", "--learning-rate",
    help="Learning rate for optimizer.",
    type=float,
    default=stylish.LEARNING_RATE,
    show_default=True
)
@click.option(
    "-b", "--batch-size",
    help="Batch size for training.",
    type=int,
    default=stylish.BATCH_SIZE,
    show_default=True
)
@click.option(
    "-e", "--epochs",
    help="Epochs to train for.",
    type=int,
    default=stylish.EPOCHS_NUMBER,
    show_default=True
)
@click.option(
    "-cw", "--content-weight",
    help="Weight of content in loss function.",
    type=float,
    default=stylish.CONTENT_WEIGHT,
    show_default=True
)
@click.option(
    "-sw", "--style-weight",
    help="Weight of style in loss function.",
    type=float,
    default=stylish.STYLE_WEIGHT,
    show_default=True
)
@click.option(
    "-tw", "--tv-weight",
    help="Weight of total variation term in loss function.",
    type=float,
    default=stylish.TV_WEIGHT,
    show_default=True
)
@click.option(
    "-o", "--output",
    help="Path to folder in which the trained model will be saved.",
    metavar="PATH",
    type=click.Path(),
)
def stylish_train(**kwargs):
    """Train a style generator model."""
    logger = stylish.logging.Logger(__name__ + ".stylish_train")

    style_path = kwargs.get("style")
    training_path = kwargs.get("training")
    vgg_path = kwargs.get("vgg")

    # Create output path.
    path = kwargs.get("output") or tempfile.gettempdir()
    name = os.path.basename(style_path.split(".", 1)[0])
    output_path = os.path.join(
        path, stylish.filesystem.sanitise_value(name, case_sensitive=False)
    )

    if os.path.isdir(output_path):
        if not click.confirm("Output path already exists, overwrite?"):
            logger.warning("Aborted!")
            return

        shutil.rmtree(output_path)

    stylish.filesystem.ensure_directory(output_path)

    # Training style generator.
    stylish.train_model(
        style_path, training_path, output_path, vgg_path,
        learning_rate=kwargs.get("learning_rate"),
        batch_size=kwargs.get("batch_size"),
        epoch_number=kwargs.get("epochs"),
        content_weight=kwargs.get("content_weight"),
        style_weight=kwargs.get("style_weight"),
        tv_weight=kwargs.get("tv_weight"),
        limit_training=kwargs.get("limit")
    )


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

    model_path = kwargs.get("model")
    input_path = kwargs.get("input")
    output_path = kwargs.get("output") or tempfile.gettempdir()

    # Ensure that the output path exist and is accessible.
    stylish.filesystem.ensure_directory(kwargs["output"])

    path = stylish.apply_model(model_path, input_path, output_path)

    logger.info("Image generated: {}".format(path))


def _download(logger, uri, path):
    """Download a file from *uri* into *path*."""
    response = requests.get(uri, allow_redirects=True, stream=True)
    if response.status_code != requests.codes.ok:
        logger.error("Unable to connect {0}".format(uri))
        response.raise_for_status()

    if os.path.exists(path):
        if not click.confirm("Path already exists, overwrite?"):
            logger.warning("Aborted!")
            return

        os.remove(path)

    iterator = response.iter_content(chunk_size=1024)
    total_length = int(response.headers.get("content-length"))

    with contextlib.ExitStack() as stack:
        progress = stack.enter_context(
            click.progressbar(iterator, length=total_length)
        )
        stream = stack.enter_context(open(path, "wb"))

        for chunk in progress:
            stream.write(chunk)
            progress.update(len(chunk))
