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


def network(input_node):
    """Apply the image transformation network.

    The last tensor of the graph will be returned. The network will be applied
    to the current Tensorflow graph.

    Example::

        >>> g = tf.Graph()
        >>> with g.as_default(), tf.Session() as session:
        ...     ...
        ...     network(input_node)

    *input_node* should be a 4-D Tensor representing a batch list of images.
    It will be the input of the network.

    """
    node = add_convolution_layer(
        input_node,
        in_channels=3,
        out_channels=32,
        kernel_size=9,
        strides=1,
        activation=True
    )
    node = add_convolution_layer(
        node,
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        strides=2,
        activation=True
    )
    node = add_convolution_layer(
        node,
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        strides=2,
        activation=True
    )

    node = add_residual_block(
        node,
        in_channels=128,
        out_channels=128,
        kernel_size=3,
        strides=1
    )
    node = add_residual_block(
        node,
        in_channels=128,
        out_channels=128,
        kernel_size=3,
        strides=1
    )
    node = add_residual_block(
        node,
        in_channels=128,
        out_channels=128,
        kernel_size=3,
        strides=1
    )
    node = add_residual_block(
        node,
        in_channels=128,
        out_channels=128,
        kernel_size=3,
        strides=1
    )
    node = add_residual_block(
        node,
        in_channels=128,
        out_channels=128,
        kernel_size=3,
        strides=1
    )

    node = add_deconvolution_layer(
        node,
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        strides=2,
        activation=True
    )
    node = add_deconvolution_layer(
        node,
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        strides=2,
        activation=True
    )
    node = add_convolution_layer(
        node,
        in_channels=32,
        out_channels=3,
        kernel_size=9,
        strides=1
    )

    output = tf.add(tf.nn.tanh(node) * 150, 255.0/2, name="output")
    return output


def add_residual_block(
    input_node, in_channels, out_channels, kernel_size, strides
):
    """Apply a residual block to the network.

    *input_node* will be the input of the block.

    *in_channels* should be the number of channels at the input of the block.

    *out_channels* should be the number of channels at the output of the block.

    *kernel_size* should be the width and height of the convolution matrix used
    within the block.

    *strides* should indicate the stride of the sliding window for each
    dimension of *input_tensor*.

    """
    node = add_convolution_layer(
        input_node,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides
    )

    node = tf.nn.relu(node)

    node = add_convolution_layer(
        node,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides
    )
    return input_node + node


def add_convolution_layer(
    input_node, in_channels, out_channels, kernel_size, strides,
    activation=False
):
    """Apply a 2-D convolution layer to the network.

    *input_node* will be the input of the layer.

    *in_channels* should be the number of channels at the input of the layer.

    *out_channels* should be the number of channels at the output of the layer.

    *kernel_size* should be the width and height of the convolution matrix used
    within the block.

    *strides* should indicate the stride of the sliding window for each
    dimension of *input_tensor*.

    *activation* should indicate whether a 'relu' tensor should be added after
    the convolution layer.

    """
    weights_shape = [kernel_size, kernel_size, in_channels, out_channels]
    weights_init = tf.Variable(
        tf.truncated_normal(weights_shape, stddev=0.1, seed=1), dtype=tf.float32
    )

    strides_shape = [1, strides, strides, 1]
    tensor = tf.nn.conv2d(
        input_node, weights_init, strides_shape, padding="SAME"
    )

    tensor = add_instance_normalization(tensor, out_channels)
    if activation:
        tensor = tf.nn.relu(tensor)

    return tensor


def add_deconvolution_layer(
    input_node, in_channels, out_channels, kernel_size, strides,
    activation=None
):
    """Apply a transposed 2-D convolution layer to the network.

    *input_node* will be the input of the layer.

    *in_channels* should be the number of channels at the input of the layer.

    *out_channels* should be the number of channels at the output of the layer.

    *kernel_size* should be the width and height of the convolution matrix used
    within the block.

    *strides* should indicate the stride of the sliding window for each
    dimension of *input_tensor*.

    *activation* should indicate whether a 'relu' tensor should be added after
    the convolution layer.

    """
    weights_shape = [kernel_size, kernel_size, in_channels, out_channels]
    weights_init = tf.Variable(
        tf.truncated_normal(weights_shape, stddev=0.1, seed=1), dtype=tf.float32
    )

    shape = tf.shape(input_node)

    strides_shape = [1, strides, strides, 1]
    new_rows = tf.multiply(shape[1], strides)
    new_columns = tf.multiply(shape[2], strides)
    new_shape = [shape[0], new_rows, new_columns, in_channels]
    tf_shape = tf.stack(new_shape)

    node = tf.nn.conv2d_transpose(
        input_node, weights_init, tf_shape, strides_shape, padding="SAME"
    )

    node = add_instance_normalization(node, in_channels)
    if activation is not None:
        node = tf.nn.relu(node)

    return node


def add_instance_normalization(input_node, channels):
    """Apply an instance normalization to the network.

    *input_node* will be the input of the layer.

    .. seealso::

        Ulyanov et al. (2017). Instance Normalization: The Missing Ingredient
        for Fast Stylization. :ref:`CoRR, abs/1607.08022
        <https://arxiv.org/abs/1607.08022>`.

    """
    mu, sigma_sq = tf.nn.moments(input_node, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros([channels]))
    scale = tf.Variable(tf.ones([channels]))
    epsilon = 1e-3
    normalized = (input_node - mu) / (sigma_sq + epsilon) ** .5
    return tf.add(tf.multiply(scale, normalized), shift)
