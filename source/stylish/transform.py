# :coding: utf-8

"""The image transformation network is a deep residual convolutional neural
network parameterized by weights.

The network body consists of five residual blocks. All non-residual
convolutional layers are followed by an instance normalization and ReLU
non-linearities with the exception of the output layer, which instead uses a
scaled "tanh" to ensure that the output image has pixels in the range [0, 255].
Other than the first and last layers which use 9 × 9 kernels, all convolutional
layers use 3 × 3 kernels.

.. seealso::

    Johnson et al. (2016). Perceptual losses for real-time style transfer and
    superresolution. :ref:`CoRR, abs/1603.08155
    <https://arxiv.org/abs/1603.08155>`.

.. seealso::

    Ulyanov et al. (2017). Instance Normalization: The Missing Ingredient for
    Fast Stylization. :ref:`CoRR, abs/1607.08022
    <https://arxiv.org/abs/1607.08022>`.

"""

import tensorflow as tf


def network(input_images):
    """Apply the image transformation network.

    The last tensor of the graph will be returned. The network will be applied
    to the current Tensorflow graph.

    Example::

        >>> g = tf.Graph()
        >>> with g.as_default(), tf.Session() as session:
        ...     ...
        ...     network(input_images)

        Tensor("tanh"),

    *input_images* should be a 4-D Tensor representing a batch list of images.
    It will be the input of the network.

    """
    conv1 = add_convolution_layer(
        input_images, channels=32, kernel_size=9, strides=1, activation=True
    )
    conv2 = add_convolution_layer(
        conv1, channels=64, kernel_size=3, strides=2, activation=True
    )
    conv3 = add_convolution_layer(
        conv2, channels=128, kernel_size=3, strides=2, activation=True
    )

    resid1 = add_residual_block(conv3, kernel_size=3)
    resid2 = add_residual_block(resid1, kernel_size=3)
    resid3 = add_residual_block(resid2, kernel_size=3)
    resid4 = add_residual_block(resid3, kernel_size=3)
    resid5 = add_residual_block(resid4, kernel_size=3)

    deconv1 = add_deconvolution_layer(
        resid5, channels=64, kernel_size=3, strides=2, activation=True
    )
    deconv2 = add_deconvolution_layer(
        deconv1, channels=32, kernel_size=3, strides=2, activation=True
    )

    conv4 = add_convolution_layer(
        deconv2, channels=3, kernel_size=9, strides=1
    )
    output = tf.nn.tanh(conv4) * 150 + 255.0/2
    return output


def add_residual_block(input_tensor, kernel_size):
    """Apply a residual block to the network.

    *input_tensor* will be the input of the block.

    *kernel_size* should be the width and height of the convolution matrix used
    within the block.

    """
    conv1 = add_convolution_layer(input_tensor, 128, kernel_size, 1)
    relu = tf.nn.relu(conv1)
    conv2 = add_convolution_layer(relu, 128, kernel_size, 1)
    return input_tensor + conv2


def add_convolution_layer(
    input_tensor, channels, kernel_size, strides, activation=None
):
    """Apply a 2-D convolution layer to the network.

    *input_tensor* will be the input of the layer.

    *channels* should be the number of channels used as an input.

    *kernel_size* should be the width and height of the convolution matrix used
    within the block.

    *strides* should indicate the stride of the sliding window for each
    dimension of *input_tensor*.

    *activation* should indicate whether a 'relu' tensor should be added after
    the convolution layer.

    """
    _channels = input_tensor.shape[-1].value
    weights_shape = [kernel_size, kernel_size, _channels, channels]
    weights_init = tf.Variable(
        tf.truncated_normal(weights_shape, stddev=0.1, seed=1), dtype=tf.float32
    )

    strides_shape = [1, strides, strides, 1]
    tensor = tf.nn.conv2d(
        input_tensor, weights_init, strides_shape, padding="SAME"
    )

    tensor = add_instance_normalization(tensor)
    if activation is not None:
        tensor = tf.nn.relu(tensor)

    return tensor


def add_deconvolution_layer(
    input_tensor, channels, kernel_size, strides, activation=None
):
    """Apply a transposed 2-D convolution layer to the network.

    *input_tensor* will be the input of the layer.

    *channels* should be the number of channels used as an output.

    *kernel_size* should be the width and height of the convolution matrix used
    within the block.

    *strides* should indicate the stride of the sliding window for each
    dimension of *input_tensor*.

    *activation* should indicate whether a 'relu' tensor should be added after
    the convolution layer.

    """
    _channels = input_tensor.shape[-1].value
    weights_shape = [kernel_size, kernel_size, channels, _channels]
    weights_init = tf.Variable(
        tf.truncated_normal(weights_shape, stddev=0.1, seed=1), dtype=tf.float32
    )

    batch_size, rows, columns, _ = [
        dimension.value for dimension in input_tensor.get_shape()
    ]
    strides_shape = [1, strides, strides, 1]
    new_rows, new_columns = int(rows * strides), int(columns * strides)
    new_shape = [batch_size, new_rows, new_columns, channels]
    tf_shape = tf.stack(new_shape)

    tensor = tf.nn.conv2d_transpose(
        input_tensor, weights_init, tf_shape, strides_shape, padding="SAME"
    )

    tensor = add_instance_normalization(tensor)
    if activation is not None:
        tensor = tf.nn.relu(tensor)

    return tensor


def add_instance_normalization(input_tensor):
    """Apply an instance normalization to the network.

    *input_tensor* will be the input of the layer.

    .. seealso::

        Ulyanov et al. (2017). Instance Normalization: The Missing Ingredient
        for Fast Stylization. :ref:`CoRR, abs/1607.08022
        <https://arxiv.org/abs/1607.08022>`.

    """
    _channels = input_tensor.shape[-1].value
    variable_shape = [_channels]
    mu, sigma_sq = tf.nn.moments(input_tensor, [1, 2], keep_dims=True)

    shift = tf.Variable(tf.zeros(variable_shape))
    scale = tf.Variable(tf.ones(variable_shape))
    epsilon = 1e-3
    normalized = (input_tensor - mu) / (sigma_sq + epsilon)**.5

    return scale * normalized + shift

