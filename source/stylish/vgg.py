# :coding: utf-8

"""Training model computation module from a :term:`Vgg19` model.

The :term:`Vgg19` model pre-trained for image classification is used as a loss
network in order to define perceptual loss functions that measure perceptual
differences in content and style between images.

The loss network remains fixed during the training process.

.. seealso::

    Johnson et al. (2016). Perceptual losses for real-time style transfer and
    superresolution. `CoRR, abs/1603.08155
    <https://arxiv.org/abs/1603.08155>`_.

.. seealso::

    Simonyan et al. (2014). Very Deep Convolutional Networks for
    Large-Scale Image Recognition. `CoRR, abs/1409.1556
    <https://arxiv.org/abs/1409.1556>`_.

    And the corresponding `Vgg19 pre-trained model
    <http://www.robots.ox.ac.uk/~vgg/research/very_deep/>`_ in the
    :term:`MatConvNet` data format.

"""

import tensorflow as tf
import numpy as np
import scipy.io

import stylish.logging


# Mean pixels value from pre-trained Vgg19 model.
VGG19_MEAN = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

#: List of layers used to extract style features with corresponding weights.
STYLE_LAYERS = [
    ("conv1_1", 0.2),
    ("conv2_1", 0.2),
    ("conv3_1", 0.2),
    ("conv4_1", 0.2),
    ("conv5_1", 0.2)
]

#: Layer used to extract the content features.
CONTENT_LAYER = "conv4_2"


def extract_mapping(path):
    """Compute and return weights and biases mapping from :term:`Vgg19` model
    *path*.

    The mapping should be returned in the form of::

        {
            "conv1_1": {
                "weight": numpy.ndarray([...]),
                "bias": numpy.ndarray([...])
            },
            "conv1_2": {
                "weight": numpy.ndarray([...]),
                "bias": numpy.ndarray([...])
            },
            ...
        }

    *path* should be the path to the :term:`Vgg19` pre-trained model in the
    :term:`MatConvNet` data format.

    .. seealso::

        http://www.vlfeat.org/matconvnet/pretrained/

    Raise :exc:`RuntimeError` if the model loaded is incorrect.

    """
    logger = stylish.logging.Logger(__name__ + ".extract_mapping")

    # All layers and index that should be extracted from the Vgg19 model.
    vgg_layers = [
        ("conv1_1", 0), ("conv1_2", 2),
        ("conv2_1", 5), ("conv2_2", 7),
        ("conv3_1", 10), ("conv3_2", 12), ("conv3_3", 14), ("conv3_4", 16),
        ("conv4_1", 19), ("conv4_2", 21), ("conv4_3", 23), ("conv4_4", 25),
        ("conv5_1", 28), ("conv5_2", 30), ("conv5_3", 32), ("conv5_4", 34)
    ]

    # Compute the mapping model.
    mapping = {}

    try:
        data = scipy.io.loadmat(path)
        layers = data["layers"]

        for name, index in vgg_layers:
            _name = layers[0][index][0][0][0][0]
            values = layers[0][index][0][0][2]

            if name != _name:
                raise RuntimeError(
                    "Layer index '{}' should be called '{}'".format(index, name)
                )

            mapping[name] = {
                "weight": values[0][0],
                "bias": values[0][1]
            }

    except Exception as error:
        raise RuntimeError("The VGG19 model is incorrect [{}]".format(error))

    logger.info(
        "Extract weights and biases from Vgg19 pre-trained model: {}"
        .format(path)
    )
    return mapping


def network(vgg_mapping, input_node):
    """Compute and return network from *mapping* with an *input_node*.

    *vgg_mapping* should gather all weight and bias matrices extracted from a
    pre-trained :term:`Vgg19` model (e.g. :func:`extract_mapping`).

    *input_node* should be a 3-D Tensor representing an image of undefined
    size with 3 channels (Red, Green and Blue). It will be the input of the
    graph model.

    """
    layer = conv2d_layer("conv1_1", vgg_mapping, input_node)
    layer = conv2d_layer("conv1_2", vgg_mapping, layer)
    layer = pool_layer("pool1", layer)

    layer = conv2d_layer("conv2_1", vgg_mapping, layer)
    layer = conv2d_layer("conv2_2", vgg_mapping, layer)
    layer = pool_layer("pool2", layer)

    layer = conv2d_layer("conv3_1", vgg_mapping, layer)
    layer = conv2d_layer("conv3_2", vgg_mapping, layer)
    layer = conv2d_layer("conv3_3", vgg_mapping, layer)
    layer = conv2d_layer("conv3_4", vgg_mapping, layer)
    layer = pool_layer("pool3", layer)

    layer = conv2d_layer("conv4_1", vgg_mapping, layer)
    layer = conv2d_layer("conv4_2", vgg_mapping, layer)
    layer = conv2d_layer("conv4_3", vgg_mapping, layer)
    layer = conv2d_layer("conv4_4", vgg_mapping, layer)
    layer = pool_layer("pool4", layer)

    layer = conv2d_layer("conv5_1", vgg_mapping, layer)
    layer = conv2d_layer("conv5_2", vgg_mapping, layer)
    layer = conv2d_layer("conv5_3", vgg_mapping, layer)
    layer = conv2d_layer("conv5_4", vgg_mapping, layer)
    layer = pool_layer("pool5", layer)
    return layer


def conv2d_layer(name, vgg_mapping, input_node):
    """Add 2D convolution layer named *name* to *mapping*.

    The layer returned should contain:

    - A `2D convolution node
      <https://www.tensorflow.org/api_docs/python/tf/nn/conv2d>`_
    - A `ReLU activation node
      <https://www.tensorflow.org/api_docs/python/tf/nn/relu>`_

    *name* should be the name of the convolution layer.

    *vgg_mapping* should gather all weight and bias matrices extracted from a
    pre-trained :term:`Vgg19` model (e.g. :func:`extract_mapping`).

    *input_node* should be a Tensor that will be set as the input of the
    convolution layer.

    Raise :exc:`KeyError` if the weight and bias matrices cannot be
    extracted from *vgg_mapping*.

    """
    logger = stylish.logging.Logger(__name__ + ".conv2d_layer")

    weight = vgg_mapping[name]["weight"]
    bias = vgg_mapping[name]["bias"]

    layer = tf.nn.conv2d(
        input_node,
        filter=tf.constant(weight),
        strides=[1, 1, 1, 1],
        padding="SAME",
    )

    layer = layer + tf.constant(np.reshape(bias, bias.size))
    layer = tf.nn.relu(layer, name=name)

    logger.debug(
        "Conv-2D layer '{}' added with ReLU activation [shape: {}]"
        .format(name, layer.shape)
    )
    return layer


def pool_layer(name, input_node):
    """Return max pooling layer named *name*.

    The layer returned should contain:

    - An `max pooling node
      <https://www.tensorflow.org/api_docs/python/tf/nn/max_pool>`_

    *name* should be the name of the max layer.

    *input_node* should be a Tensor that will be set as the input of the
    max layer.

    """
    logger = stylish.logging.Logger(__name__ + ".pool_layer")

    layer = tf.nn.max_pool(
        input_node,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        name=name
    )

    logger.debug(
        "Max Pool layer '{}' added [shape: {}]".format(name, layer.shape)
    )
    return layer
