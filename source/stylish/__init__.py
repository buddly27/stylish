# :coding: utf-8

import os

import stylish.logging
import stylish.filesystem
import stylish.vgg
import stylish.core
from stylish._version import __version__


def create_model(
    training_path, style_path, output_path, vgg_path, learning_rate=None,
    batch_size=None, batch_shape=None, epoch_number=None, content_weight=None,
    style_weight=None, tv_weight=None, content_layer=None, style_layers=None,
    limit_training=None
):
    """Train a style generator model based on an image and a dataset folder

    Usage example::

        >>> create_model(
        ...    "/path/to/training_data/",
        ...    "/path/to/style_image.jpg",
        ...    "/path/to/output_model/",
        ...    "/path/to/vgg_model.mat"
        ... )

    :param training_path: training dataset folder.

    :param style_path: path to an image from which the style features
        will be extracted.

    :param output_path: path where the trained model and logs should be saved

    :param vgg_path: path to the :term:`Vgg19` pre-trained model in the
        :term:`MatConvNet` data format.

    :param learning_rate: :term:`Learning Rate` value to train the model.
        Default is :data:`stylish.core.LEARNING_RATE`.

    :param batch_size: number of images to use in one training iteration.
        Default is :data:`stylish.core.BATCH_SIZE`.

    :param batch_shape: shape used for each images within training dataset.
        Default is :data:`stylish.core.BATCH_SHAPE`.

    :param epoch_number: number of time that model should be trained against
        *training_images*. Default is :data:`stylish.core.EPOCHS_NUMBER`.

    :param content_weight: weight of the content feature cost. Default is
        :data:`stylish.core.CONTENT_WEIGHT`.

    :param style_weight: weight of the style feature cost. Default is
        :data:`stylish.core.STYLE_WEIGHT`.

    :param tv_weight: weight of the total variation cost. Default is
        :data:`stylish.core.TV_WEIGHT`.

    :param content_layer: Layer name from pre-trained :term:`Vgg19` model
        used to extract the content information. Default is
        :data:`stylish.vgg.CONTENT_LAYER`.

    :param style_layers: Layer names from pre-trained :term:`Vgg19` model
        used to extract the style information with corresponding weights.
        Default is :data:`stylish.vgg.STYLE_LAYERS`.

    :param limit_training: maximum number of files to use from the training
        dataset folder. By default, all files from the training dataset folder
        are used.

    :return: None

    """
    logger = stylish.logging.Logger(__name__ + ".train_model")
    logger.info("Train model for style image: {}".format(style_path))

    # Identify output model path
    model_path = os.path.join(output_path, "model")
    logger.info("Model will be exported in {}".format(model_path))

    # Identify output log path (to view graph with Tensorboard)
    log_path = os.path.join(output_path, "log")
    logger.info("Logs will be exported in {}".format(log_path))

    # Extract weight and bias from pre-trained Vgg19 mapping.
    vgg_mapping = stylish.vgg.extract_mapping(vgg_path)

    # Extract targeted images for training.
    logger.info("Extract content images from '{}'".format(training_path))
    training_images = stylish.filesystem.fetch_images(
        training_path, limit=limit_training
    )
    logger.info("{} content image(s) found.".format(len(training_images)))

    # Pre-compute style features.
    style_mapping = stylish.core.extract_style_from_path(
        style_path, vgg_mapping, style_layers or stylish.vgg.STYLE_LAYERS
    )

    stylish.core.train_model(
        training_images, style_mapping, vgg_mapping, model_path, log_path,
        learning_rate=learning_rate or stylish.core.LEARNING_RATE,
        batch_size=batch_size or stylish.core.BATCH_SIZE,
        batch_shape=batch_shape or stylish.core.BATCH_SHAPE,
        epoch_number=epoch_number or stylish.core.EPOCHS_NUMBER,
        content_weight=content_weight or stylish.core.CONTENT_WEIGHT,
        style_weight=style_weight or stylish.core.STYLE_WEIGHT,
        tv_weight=tv_weight or stylish.core.TV_WEIGHT,
        content_layer=content_layer or stylish.vgg.CONTENT_LAYER,
        style_layers=style_layers or [
            name for name, _ in stylish.vgg.STYLE_LAYERS
        ]
    )


def apply_model(model_path, input_path, output_path):
    """Apply style generator model to a new image.

    Usage example::

        >>> apply_model(
        ...    "/path/to/saved_model/",
        ...    "/path/to/image.jpg",
        ...    "/path/to/output/"
        ... )

        /path/to/output/image.jpg

    :param model_path: path to trained model saved.

    :param input_path: path to image to inferred model to.

    :param output_path: path folder to save image output.

    """
    logger = stylish.logging.Logger(__name__ + ".apply_model")
    logger.info("Apply style generator model.")

    return stylish.core.infer_model(model_path, input_path, output_path)
