# :coding: utf-8

import logging

import tensorflow as tf
import numpy as np
import scipy.io


#: List of layers from the Vgg19 model file.
VGG19_LAYERS = (
    "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1", "conv2_1", "relu2_1",
    "conv2_2", "relu2_2", "pool2", "conv3_1", "relu3_1", "conv3_2", "relu3_2",
    "conv3_3", "relu3_3", "conv3_4", "relu3_4", "pool3", "conv4_1", "relu4_1",
    "conv4_2", "relu4_2", "conv4_3", "relu4_3", "conv4_4", "relu4_4", "pool4",
    "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "conv5_4",
    "relu5_4"
)

#: List of layers used to extract the style features from an image.
STYLE_LAYERS = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")

#: Layers use to extract the content features from an image.
CONTENT_LAYER = "relu4_2"


def extract_data(model_path):
    """Return layers and mean pixel arrays from Vgg19 model *data_path*.

    *model_path* should be the path to the Vgg19 pre-trained model in the
    MatConvNet data format.

    .. seealso::

        http://www.vlfeat.org/matconvnet/pretrained/

    Raise :exc:`ValueError` if the model loaded is incorrect.

    """
    logging.info("Extract data from: {!r}".format(model_path))

    data = scipy.io.loadmat(model_path)
    if not all(i in data for i in ("layers", "classes", "normalization")):
        raise ValueError("You VGG19 model file is incorrect.")

    mean = data["normalization"][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    layers = data["layers"][0]
    return layers, mean_pixel


def compute_network(input_image, layers):
    """Return network from *input_image* tensor and Vgg19 weights array.

    *input_image* should be a 3-D Tensor representing an image of undefined size
    with 3 channels (Red, Green and Blue).

    *layers* should be an array of layers :func:`extracted <extract_data>` from
    the Vgg19 model file.

    """
    logging.debug("Compute network.")

    mapping = {}

    current = input_image

    for index, name in enumerate(VGG19_LAYERS):
        layer_type = name[:4]

        if layer_type == "conv":
            weights, bias = layers[index][0][0][0][0]

            # The weight array extracted from the MatConvNet data format
            # must be transposed to match the tensorflow format.
            #
            # from: [width, height, in_channels, out_channels]
            # to: [height, width, in_channels, out_channels]
            weights = np.transpose(weights, (1, 0, 2, 3))
            bias = bias.reshape(-1)

            layer = tf.nn.conv2d(
                current, tf.constant(weights),
                strides=(1, 1, 1, 1),
                padding="SAME"
            )
            current = tf.nn.bias_add(layer, bias)

        elif layer_type == "relu":
            current = tf.nn.relu(current)

        elif layer_type == "pool":
            current = tf.nn.max_pool(
                current,
                ksize=(1, 2, 2, 1),
                strides=(1, 2, 2, 1),
                padding="SAME"
            )

        mapping[name] = current

    return mapping
