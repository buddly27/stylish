# :coding: utf-8

import os

import stylish.logging
import stylish.filesystem
import stylish.vgg
import stylish.core
from stylish._version import __version__


def transform_image(
    path, style_path, output_path, vgg_path, iterations=None,
    learning_rate=None, content_weight=None, style_weight=None, tv_weight=None,
    content_layer=None, style_layers=None, log_path=None
):
    """Generate new image from *path* with style from another image.

    Usage example::

        >>> transform_image(
        ...    "/path/to/image.jpg",
        ...    "/path/to/style_image.jpg",
        ...    "/path/to/output_image/",
        ...    "/path/to/vgg_model.mat"
        ... )

    :param path: path to the image to transform.

    :param style_path: path to an image from which the style features
        will be extracted.

    :param output_path: path where the transformed image will be generated.

    :param vgg_path: path to the :term:`Vgg19` pre-trained model in the
        :term:`MatConvNet` data format.

    :param iterations: number of time that image should be trained against
        *style_path*. Default is :data:`stylish.core.ITERATIONS_NUMBER`.

    :param learning_rate: :term:`Learning Rate` value to train the model.
        Default is :data:`stylish.core.LEARNING_RATE`.

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

    :param log_path: path to extract the log information. Default is the
        temporary directory.

    :return: path to transformed image.

    .. note::

        `Lists of all image formats currently supported
        <https://imageio.readthedocs.io/en/stable/formats.html>`_

    """
    logger = stylish.logging.Logger(__name__ + ".transform_image")
    logger.info("Create new image with style from {}".format(style_path))
    logger.info("Logs path: {}".format(log_path))

    # Compute output image path.
    base_name, extension = os.path.splitext(path)
    base_name = os.path.basename(base_name)
    output_image = os.path.join(output_path, base_name + extension)

    # Extract weight and bias from pre-trained Vgg19 mapping.
    vgg_mapping = stylish.vgg.extract_mapping(vgg_path)

    # Load image from path.
    image = stylish.filesystem.load_image(path)

    # Pre-compute style features.
    style_mapping = stylish.core.extract_style_from_path(
        style_path, vgg_mapping, style_layers or stylish.vgg.STYLE_LAYERS,
        image_size=image.shape
    )

    transformed_image = stylish.core.optimize_image(
        image, style_mapping, vgg_mapping, log_path, iterations,
        learning_rate=learning_rate, content_weight=content_weight,
        style_weight=style_weight, tv_weight=tv_weight,
        content_layer=content_layer or stylish.vgg.CONTENT_LAYER,
        style_layers=style_layers or [
            name for name, _ in stylish.vgg.STYLE_LAYERS
        ]
    )

    stylish.filesystem.save_image(transformed_image, output_image)
    return output_image


def create_model(
    training_path, style_path, output_path, vgg_path, learning_rate=None,
    batch_size=None, batch_shape=None, epoch_number=None, content_weight=None,
    style_weight=None, tv_weight=None, content_layer=None, style_layers=None,
    limit_training=None, log_path=None
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

    :param log_path: path to extract the log information. Default is the
        temporary directory.

    :return: None

    .. note::

        `Lists of all image formats currently supported
        <https://imageio.readthedocs.io/en/stable/formats.html>`_

    """
    logger = stylish.logging.Logger(__name__ + ".create_model")
    logger.info("Create model to apply style from {}".format(style_path))
    logger.info("Model path: {}".format(output_path))
    logger.info("Logs path: {}".format(log_path))

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

    stylish.core.optimize_model(
        training_images, style_mapping, vgg_mapping, output_path, log_path,
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

    :return: path to transformed image.

    .. note::

        `Lists of all image formats currently supported
        <https://imageio.readthedocs.io/en/stable/formats.html>`_

    """
    logger = stylish.logging.Logger(__name__ + ".apply_model")
    logger.info("Apply model to image {}".format(input_path))

    # Compute output image path.
    base_name, extension = os.path.splitext(input_path)
    base_name = os.path.basename(base_name)
    output_image = os.path.join(output_path, base_name + extension)

    # Infer model and save image.
    image = stylish.core.infer_model(model_path, input_path)
    stylish.filesystem.save_image(image, output_image)

    return output_image
