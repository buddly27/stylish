# :coding: utf-8

import os
import logging
import operator
import functools
import time
import uuid

import tensorflow as tf
import numpy as np

import stylish.vgg
import stylish.transform
import stylish.filesystem


BATCH_SIZE = 4
CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2
LEARNING_RATE = 1e-3
EPOCHS_NUMBER = 2


def extract_model(
    style_target, content_targets, layers, mean_pixel, model_path
):
    """Train and return style generator model path.

    *style_target* should be the path to an image from which the style features
    should be extracted.

    *content_targets* should be the list of image paths from which the content
    features should be extracted.

    *layers* should be an array of layers :func:`extracted
    <stylish.vgg.extract_data>` from the Vgg19 model file.

    *mean_pixel* should be an array of three mean pixel values for the Red,
    Green and Blue channels :func:`extracted <stylish.vgg.extract_data>` from
    the Vgg19 model file.

    *model_path* should be the path where the trained model should be saved.

    """
    logging.info("Train style generator model.")

    # Pre-compute style feature map.
    style_features = compute_style_features(style_target, layers, mean_pixel)

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

        # Start training.

        logging.info("Start training.")

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        train_size = len(content_targets)

        for epoch in range(EPOCHS_NUMBER):
            logging.debug("Start epoch #{}.".format(epoch))

            start_time = time.time()

            for iteration in range(train_size // BATCH_SIZE):
                logging.debug("Start processing batch #{}.".format(iteration))
                _start_time = time.time()

                x_batch = get_next_batch(
                    iteration, content_targets, BATCH_SIZE, batch_shape
                )

                session.run(training_op, feed_dict={content_image: x_batch})

                _end_time = time.time()
                logging.debug(
                    "Batch #{} processed [time: {}].".format(
                        iteration, _end_time - _start_time
                    )
                )

            end_time = time.time()
            logging.debug(
                "Epoch #{} processed [time: {}].".format(
                    epoch, end_time - start_time
                )
            )

            model_name = "style_model_{}".format(uuid.uuid4())
            return saver.save(session, os.path.join(model_path, model_name))


def compute_style_features(style_target, layers, mean_pixel):
    """Return computed style features map from *image_matrix*.

    The style feature map will be used to penalize the predicted image when it
    deviates from the style (colors, textures, common patterns, etc.).

    *style_target* should be the path to an image from which the style features
    should be extracted.

    *layers* should be an array of layers :func:`extracted
    <stylish.vgg.extract_data>` from the Vgg19 model file.

    *mean_pixel* should be an array of three mean pixel values for the Red,
    Green and Blue channels :func:`extracted <stylish.vgg.extract_data>` from
    the Vgg19 model file.

    """
    logging.info("Compute style features for image.")

    # Extract image matrix from image.
    image_matrix = stylish.filesystem.load_image(style_target)

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
            logging.debug("Start processing style layer {!r}.".format(layer))

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
    """Compute loss ratio from *predictions*.

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
        2 * tf.nn.l2_loss(
            network[stylish.vgg.CONTENT_LAYER] - content_features
        ) / content_size
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


def get_next_batch(iteration, content_targets, batch_size, batch_shape):
    """Return Numpy array with image matrices according to *iteration* index.

    *iteration* should be an integer specifying the current portion of the
    images to return.

    *content_targets* should be the list of image paths from which the content
    features should be extracted.

    *batch_size* should be the size of the image list to return.

    *batch_shape* should be indicate the dimensions in which each image should
    be resized to.

    """
    current = iteration * batch_size
    step = current + batch_size

    x_batch = np.zeros(batch_shape, dtype=np.float32)

    # Extract and resize images from training data.
    for index, image_path in enumerate(content_targets[current:step]):
        x_batch[index] = stylish.filesystem.load_image(
            image_path, image_size=batch_shape[1:]
        ).astype(np.float32)

    return x_batch


def _extract_tensor_size(tensor):
    """Extract dimension from *tensor*."""
    return functools.reduce(
        operator.mul, (dim.value for dim in tensor.get_shape()[1:]), 1
    )
