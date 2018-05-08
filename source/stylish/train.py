# :coding: utf-8

import logging
import operator
import functools

import tensorflow as tf
import numpy as np

import stylish.vgg
import stylish.transform


BATCH_SIZE = 4
CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2
LEARNING_RATE = 1e-3


def execute(image_matrix, layers, mean_pixel):
    """Train style generator model from *image_matrix*.

    *image_matrix* should be a 3-D Numpy array representing an image with 3
    channels (Red, Green and Blue).

    *layers* should be an array of layers :func:`extracted
    <stylish.vgg.extract_data>` from the Vgg19 model file.

    *mean_pixel* should be an array of three mean pixel values for the Red,
    Green and Blue channels :func:`extracted <stylish.vgg.extract_data>` from
    the Vgg19 model file.

    """
    logging.info("Train model from image.")

    # Pre-compute style feature map.
    style_features = compute_style_features(image_matrix, layers, mean_pixel)

    # Initiate a default graph.
    graph = tf.Graph()

    # Initiate the batch shape.
    batch_shape = (BATCH_SIZE, 256, 256, 3)

    with graph.as_default(), tf.Session() as session:
        content_image = tf.placeholder(
            tf.float32, shape=batch_shape, name="content_image"
        )
        normalized_image = content_image - mean_pixel
        loss_network = stylish.vgg.extract_network(normalized_image, layers)

        content_features = loss_network[stylish.vgg.CONTENT_LAYER]
        predictions = stylish.transform.network(content_image/255.0)

        loss = compute_loss_ratio(
            predictions, batch_shape, style_features, content_features,
            layers, mean_pixel
        )

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        training_op = optimizer.minimize(loss)


def compute_style_features(image_matrix, layers, mean_pixel):
    """Return computed style features map from *image_matrix*.

    The style feature map will be used to penalize the predicted image when it
    deviates from the style (colors, textures, common patterns, etc.).

    *image_matrix* should be a 3-D Numpy array representing an image of
    undefined size with 3 channels (Red, Green and Blue).

    *layers* should be an array of layers :func:`extracted
    <stylish.vgg.extract_data>` from the Vgg19 model file.

    *mean_pixel* should be an array of three mean pixel values for the Red,
    Green and Blue channels :func:`extracted <stylish.vgg.extract_data>` from
    the Vgg19 model file.

    """
    logging.info("Compute style features for image.")

    # Initiate a default graph.
    graph = tf.Graph()

    # Initiate the shape of a 4-D Tensor for a list of images.
    image_shape = (1,) + image_matrix.shape

    # Initiate the style features.
    style_features = {}

    with graph.as_default(), tf.Session() as session:
        style_image = tf.placeholder(
            tf.float32, shape=image_shape, name="style_image"
        )
        normalized_image = style_image - mean_pixel
        loss_network = stylish.vgg.extract_network(normalized_image, layers)

        # Initiate input as a list of images.
        images = np.array([image_matrix])

        for layer in stylish.vgg.STYLE_LAYERS:
            logging.debug("Process layer {!r}.".format(layer))
            features = session.run(
                loss_network[layer], feed_dict={style_image: images}
            )
            logging.debug("Layer {!r} processed.".format(layer))

            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    return style_features


def compute_loss_ratio(
    predictions, batch_shape, style_features, content_features,
    layers, mean_pixel
):
    """Compute the loss ratio from *predictions*.

    *predictions* should be the output tensor of the
    :func:`transformation network <stylish.transform.network>`.

    *batch_shape* should be the shape of the 4-D input image list tensor.

    *style_features* should be the style features map :func:`extracted
    <compute_style_features>`.

    *content_features* should be the content features map extracted.

    *layers* should be an array of layers :func:`extracted
    <stylish.vgg.extract_data>` from the Vgg19 model file.

    *mean_pixel* should be an array of three mean pixel values for the Red,
    Green and Blue channels :func:`extracted <stylish.vgg.extract_data>` from
    the Vgg19 model file.

    """
    normalized_predictions = predictions - mean_pixel

    network = stylish.vgg.extract_network(normalized_predictions, layers)

    # Compute feature reconstruction loss from content feature map.

    logging.info("Compute feature reconstruction loss ratio.")

    content_size = _extract_tensor_size(content_features) * BATCH_SIZE
    content_loss = CONTENT_WEIGHT * (
        2 * tf.nn.l2_loss(content_features - content_features) / content_size
    )

    # Compute style reconstruction loss from style features map.

    logging.info("Compute style reconstruction loss ratio.")

    style_losses = []

    for style_layer in stylish.vgg.STYLE_LAYERS:
        layer = network[style_layer]
        batch, height, width, filters = [
            dimension.value for dimension in layer.get_shape()
        ]

        size = height * width * filters
        feats = tf.reshape(layer, (batch, height * width, filters))
        feats_transposed = tf.transpose(feats, perm=[0, 2, 1])
        grams = tf.matmul(feats_transposed, feats) / size
        style_gram = style_features[style_layer]
        style_losses.append(
            2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size
        )

    style_loss = (
        STYLE_WEIGHT * functools.reduce(tf.add, style_losses) / BATCH_SIZE
    )

    # Compute total variation de-noising.

    logging.info("Compute total variation loss ratio.")

    tv_y_size = _extract_tensor_size(predictions[:, 1:, :, :])
    tv_x_size = _extract_tensor_size(predictions[:, :, 1:, :])

    y_tv = tf.nn.l2_loss(
        predictions[:, 1:, :, :] - predictions[:, :batch_shape[1] - 1, :, :]
    )
    x_tv = tf.nn.l2_loss(
        predictions[:, :, 1:, :] - predictions[:, :, :batch_shape[2] - 1, :]
    )
    total_variation_loss = (
        TV_WEIGHT * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / BATCH_SIZE
    )

    return content_loss + style_loss + total_variation_loss


def _extract_tensor_size(tensor):
    """Extract dimension from *tensor*."""
    return functools.reduce(
        operator.mul, (dim.value for dim in tensor.get_shape()[1:]), 1
    )
