# :coding: utf-8

import logging

import tensorflow as tf
import numpy as np

import stylish.model


def execute(image_matrix, layers, mean_pixel):
    """Train model which transfer the style of *image_matrix* to other targets.

    *image_matrix* should be a 3-D Tensor representing an image of undefined size
    with 3 channels (Red, Green and Blue).

    """
    logging.info("Train model from image.")

    style_features = compute_style_features(image_matrix, layers, mean_pixel)


def compute_style_features(image_matrix, layers, mean_pixel):
    """Return mapping of extracted style features from *image_matrix*.

    *image_matrix* should be a 3-D Numpy array representing an image of
    undefined size with 3 channels (Red, Green and Blue).

    *layers* should be an array of layers :func:`extracted <extract_data>` from
    the Vgg19 model file.

    *mean_pixel* should be an array of three mean pixel values for the Red,
    Green and Blue channels :func:`extracted <extract_data>` from the Vgg19
    model file.

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
        network = stylish.model.compute_network(normalized_image, layers)

        # Initiate input as a list of images.
        images = np.array([image_matrix])

        for layer in stylish.model.STYLE_LAYERS:
            logging.debug("Process layer {!r}.".format(layer))
            features = session.run(
                network[layer], feed_dict={style_image: images}
            )
            logging.debug("Layer {!r} processed.".format(layer))

            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    return style_features
