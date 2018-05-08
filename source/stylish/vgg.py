# :coding: utf-8

"""The Vgg19 model pre-trained for image classification is used as a loss
network in order to define perceptual loss functions that measure perceptual
differences in content and style between images.

The loss network remains fixed during the training process.

.. seealso::

    Johnson et al. (2016). Perceptual losses for real-time style transfer and
    superresolution. :ref:`CoRR, abs/1603.08155
    <https://arxiv.org/abs/1603.08155>`.

.. seealso::

    Simonyan et al. (2014). Very Deep Convolutional Networks for
    Large-Scale Image Recognition. :ref:`CoRR, abs/1409.1556
    <https://arxiv.org/abs/1409.1556>`.

    And the corresponding :ref:`Vgg19 pre-trained model
    <http://www.robots.ox.ac.uk/~vgg/research/very_deep/>` in the MatConvNet
    data format.

"""

import logging

import tensorflow as tf
import numpy as np
import scipy.io


#: List of layers which constitute the loss network.
_LAYERS = (
    "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",

    "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",

    "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "conv3_4",
    "relu3_4", "pool3",

    "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "conv4_4",
    "relu4_4", "pool4",

    "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "conv5_4",
    "relu5_4"
)

#: List of layers used to extract the style features.
STYLE_LAYERS = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")

#: Layer used to extract the content features.
CONTENT_LAYER = "relu4_2"


def extract_data(path):
    """Return layers and mean pixel arrays from Vgg19 model *data_path*.

    *path* should be the path to the Vgg19 pre-trained model in the
    MatConvNet data format.

    .. seealso::

        http://www.vlfeat.org/matconvnet/pretrained/

    Raise :exc:`ValueError` if the model loaded is incorrect.

    """
    logging.info("Extract data from: {!r}".format(path))

    data = scipy.io.loadmat(path)
    if not all(i in data for i in ("layers", "classes", "normalization")):
        raise ValueError("You VGG19 model file is incorrect.")

    mean = data["normalization"][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    layers = data["layers"][0]
    return layers, mean_pixel


def extract_network(input_image, layers):
    """Extract and apply the network from Vgg19 *layers*.

    A mapping with all nodes from the network is returned. The network will be
    applied to the current Tensorflow graph.

    Example::

        >>> g = tf.Graph()
        >>> with g.as_default(), tf.Session() as session:
        ...     ...
        ...     extract_network(input_image, layers)

        {
            "conv1_1": Tensor("conv2d"),
            "relu1_1": Tensor("relu"),
            "conv1_2": Tensor("conv2d"),
            "relu1_2": Tensor("relu"),
            ...
        }

    *input_image* should be a 3-D Tensor representing an image of undefined size
    with 3 channels (Red, Green and Blue). It will be the input of the network.

    *layers* should be an array of layers :func:`extracted <extract_data>` from
    the Vgg19 model file.

    """
    logging.debug("Extract network from Vgg19 pre-trained model.")

    mapping = {}

    current = input_image

    for index, name in enumerate(_LAYERS):
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
