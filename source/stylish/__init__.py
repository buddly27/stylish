# :coding: utf-8

import os
import time
import contextlib

import tensorflow as tf
import numpy as np

import stylish.logging
import stylish.filesystem
import stylish.vgg
import stylish.transform
from stylish._version import __version__


#: Default batch size used for training.
BATCH_SIZE = 4

#: Default epoch number used for training.
EPOCHS_NUMBER = 2

#: Default weight of the content for the loss computation.
CONTENT_WEIGHT = 7.5

#: Default weight of the style for the loss computation.
STYLE_WEIGHT = 100.0

#: Default weight of the total variation term for the loss computation.
TV_WEIGHT = 200.0

#: Default :term:`Learning Rate`.
LEARNING_RATE = 1e-3


def train_model(
    style_path, training_path, output_path, vgg_path,
    learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
    epoch_number=EPOCHS_NUMBER, content_weight=CONTENT_WEIGHT,
    style_weight=STYLE_WEIGHT, tv_weight=TV_WEIGHT, limit_training=None
):
    """Train a style generator model for *style_path* on *training_path*.

    The training duration can vary depending on the :term:`Hyperparameters
    <Hyperparameter>` specified (epoch number, batch size, etc.), the power
    of your workstation and the number of images in the training data.

    Usage example::

        >>> train_model(
        ...    "/path/to/style_image.jpg",
        ...    "/path/to/training_data/",
        ...    "/path/to/output_model/",
        ...    "/path/to/vgg_model.mat"
        ... )

    *style_path* should be the path to an image from which the style features
    will be extracted.

    *training_path* should be the training dataset folder.

    *output_path* should be the path where the trained model should be saved.

    *vgg_path* should be the path to the :term:`Vgg19` pre-trained model in the
    :term:`MatConvNet` data format.

    *learning_rate* should indicate the :term:`Learning Rate` to minimize the
    loss. Default is :data:`LEARNING_RATE`.

    *batch_size* should indicate the number of training examples utilized in one
    iteration. Default is :data:`BATCH_SIZE`.

    *epoch_number* should indicate the number of time that the *training data*
    should be trained. Default is :data:`EPOCHS_NUMBER`.

    *content_weight* should indicate the weight of the content for the loss
    computation. Default is :data:`CONTENT_WEIGHT`.

    *style_weight* should indicate the weight of the style for the loss
    computation. Default is :data:`STYLE_WEIGHT`.

    *tv_weight* should indicate the weight of the total variation term for the
    loss computation. Default is :data:`TV_WEIGHT`.

    *limit_training* should be the maximum number of files to use from the
    training dataset folder. By default, all files from the training dataset
    folder are used.

    """
    logger = stylish.logging.Logger(__name__ + ".train_model")
    logger.info("Train model for style image: {}".format(style_path))

    # Identify output model path
    output_model = os.path.join(output_path, "model")
    logger.info("Model will be exported in {}".format(output_model))

    # Identify output log path (to view graph with Tensorboard)
    output_log = os.path.join(output_path, "log")
    logger.info("Log will be exported in {}".format(output_log))

    # Identify output log path (to view graph with Tensorboard)
    output_checkpoint = os.path.join(output_path, "checkpoints")
    logger.info("Checkpoints will be exported in {}".format(output_checkpoint))

    # Extract weight and bias from pre-trained Vgg19 mapping.
    vgg_mapping = stylish.vgg.extract_mapping(vgg_path)

    # Extract targeted images for training.
    logger.info("Extract content images from '{}'".format(training_path))
    training_data = stylish.filesystem.fetch_images(
        training_path, limit=limit_training
    )
    logger.info("{} content image(s) found.".format(len(training_data)))

    # Pre-compute style features.
    with create_session() as session:
        style_feature = compute_style_feature(session, style_path, vgg_mapping)

    with create_session() as session:
        input_node = tf.placeholder(
            tf.float32, shape=(None, None, None, None), name="input"
        )

        # Normalize input.
        _input_node = input_node - stylish.vgg.VGG19_MEAN

        # Build main network.
        stylish.vgg.network(vgg_mapping, _input_node)
        output_node = stylish.transform.network(_input_node/255.0)

        # Build loss network.
        loss_mapping = compute_loss(
            session, output_node, style_feature, vgg_mapping,
            batch_size=batch_size, content_weight=content_weight,
            style_weight=style_weight, tv_weight=tv_weight,

        )

        # Apply optimizer to attempt to reduce the loss.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_node = optimizer.minimize(loss_mapping["total"])

        # Start training.
        logger.info("Start training.")

        # Train the network on training data
        optimize(
            session, training_node, training_data, input_node, loss_mapping,
            output_log, output_checkpoint, batch_size=batch_size,
            epoch_number=epoch_number
        )

        # Save model.
        input_info = tf.saved_model.utils.build_tensor_info(input_node)
        output_info = tf.saved_model.utils.build_tensor_info(output_node)

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"input": input_info},
            outputs={"output": output_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        builder = tf.saved_model.builder.SavedModelBuilder(output_model)
        builder.add_meta_graph_and_variables(
            session, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={"predict_images": signature},

        )
        builder.save()


def apply_model(model_path, input_path, output_path):
    """Apply style generator *model_path* for input image.

    Return path to image generated.

    Usage example::

        >>> apply_model(
        ...    "/path/to/saved_model/",
        ...    "/path/to/input_image.jpg",
        ...    "/path/to/output/"
        ... )

    *model_path* should be the path to a :term:`Tensorflow` model path that has
    been :func:`trained <train_model>` on an other image to extract its style.

    *input_path* should be the path to an image to apply the *model_path* to.

    *output_path* should be the folder where the output image should be saved.

    """
    logger = stylish.logging.Logger(__name__ + ".apply_model")
    logger.info("Apply style generator model.")

    # Extract image matrix from input image.
    image_matrix = stylish.filesystem.load_image(input_path)

    # Compute output image path.
    _input_image, _ = os.path.splitext(input_path)
    output_image = os.path.join(
        output_path, "{}.jpg".format(os.path.basename(_input_image))
    )

    with create_session() as session:
        graph = tf.get_default_graph()

        tf.saved_model.loader.load(session, ["serve"], model_path)
        input_node = graph.get_tensor_by_name("input:0")
        output_node = graph.get_tensor_by_name("output:0")

        start_time = time.time()

        predictions = session.run(
            output_node, feed_dict={input_node: np.array([image_matrix])}
        )
        stylish.filesystem.save_image(predictions[0], output_image)

        end_time = time.time()
        logger.info(
            "Image transformed: {} [time={}]".format(
                output_image, end_time - start_time
            )
        )

        return output_image


@contextlib.contextmanager
def create_session():
    """Create a :term:`Tensorflow` session and reset the default graph.

    Should be used as follows::

        >>> with create_session() as session:
            ...

    """
    tf.reset_default_graph()

    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    session = tf.Session(config=soft_config)

    try:
        yield session
    finally:
        session.close()


def compute_style_feature(session, path, vgg_mapping):
    """Return computed style features mapping from image *path*.

    The style feature map will be used to penalize the predicted image when it
    deviates from the style (colors, textures, common patterns, etc.).

    Usage example::

        >>> compute_style_feature(session, path, vgg_mapping)

        {
            "conv1_1": numpy.array([...]),
            "conv2_1": numpy.array([...]),
            "conv3_1": numpy.array([...]),
            "conv4_1": numpy.array([...]),
            "conv5_1": numpy.array([...])
        }

    *session* should be a :term:`Tensorflow` session.

    *path* should be the path to an image from which the style features will be
    extracted.

    *vgg_mapping* should gather all weight and bias matrices extracted from a
    pre-trained :term:`Vgg19` model (e.g. :func:`extract_mapping`).

    """
    logger = stylish.logging.Logger(__name__ + ".compute_style_feature")
    logger.info("Extract style feature mapping from path: {}".format(path))

    # Extract image matrix from image.
    image_matrix = stylish.filesystem.load_image(path)

    # Initiate the shape of a 4-D Tensor for a list of images.
    image_shape = (1,) + image_matrix.shape

    # Initiate the style features.
    style_feature = {}

    with tf.name_scope("style_feature"):
        input_node = tf.placeholder(
            tf.float32, shape=image_shape, name="input"
        )
        _input_node = input_node - stylish.vgg.VGG19_MEAN

        stylish.vgg.network(vgg_mapping, _input_node)

    # Initiate input as a list of images.
    images = np.array([image_matrix])

    for layer_name, weight in stylish.vgg.STYLE_LAYERS:
        logger.debug("Processing style layer '{}'".format(layer_name))

        graph = tf.get_default_graph()
        layer_node = graph.get_tensor_by_name(
            "style_feature/{}:0".format(layer_name)
        )

        # Run session on style layer.
        features = session.run(layer_node, feed_dict={input_node: images})
        logger.debug("Layer '{}' processed.".format(layer_name))

        features = np.reshape(features, (-1, features.shape[3]))
        gram = np.matmul(features.T, features) / features.size
        style_feature[layer_name] = gram * weight

    return style_feature


def compute_loss(
    session, input_node, style_features, vgg_mapping,
    batch_size=BATCH_SIZE, content_weight=CONTENT_WEIGHT,
    style_weight=STYLE_WEIGHT, tv_weight=TV_WEIGHT,
):
    """Create loss network from *input_node*.

    Return a mapping with the content loss, the style loss, the total variation
    loss and the total loss nodes.

    Usage example::

        >>> compute_loss(session, input_node, style_features, vgg_mapping)

        {
            "total": tf.Tensor(...),
            "content": tf.Tensor(...),
            "style": tf.Tensor(...),
            "total_variation": tf.Tensor(...)
        }

    *session* should be a :term:`Tensorflow` session.

    *input_node* should be the output tensor of the main graph.

    *style_features* should be the style features map :func:`extracted
    <compute_style_features>`.

    *vgg_mapping* should gather all weight and bias matrices extracted from a
    pre-trained :term:`Vgg19` model (e.g. :func:`extract_mapping`).

    *batch_size* should indicate the number of training examples utilized in one
    iteration. Default is :data:`BATCH_SIZE`.

    *content_weight* should indicate the weight of the content. Default is
    :data:`CONTENT_WEIGHT`.

    *style_weight* should indicate the weight of the style. Default is
    :data:`STYLE_WEIGHT`.

    *tv_weight* should indicate the weight of the total variation term. Default
    is :data:`TV_WEIGHT`.

    """
    logger = stylish.logging.Logger(__name__ + ".compute_loss")

    # Initiate the batch shape.
    batch_shape = (batch_size, 256, 256, 3)

    # Normalize predicted output.
    _output_node = input_node - stylish.vgg.VGG19_MEAN

    # Fetch content layer from main graph.
    content_layer = session.graph.get_tensor_by_name(
        "{}:0".format(stylish.vgg.CONTENT_LAYER)
    )

    # 1. Compute content loss.
    logger.info("Compute feature reconstruction loss ratio.")

    with tf.name_scope("loss_network"):
        stylish.vgg.network(vgg_mapping, _output_node)

    with tf.name_scope("content_loss"):
        content_shape = tf.cast(tf.shape(content_layer), tf.float32)
        content_size = tf.reduce_prod(content_shape[1:]) * batch_size
        _content_layer = session.graph.get_tensor_by_name(
            "loss_network/{}:0".format(stylish.vgg.CONTENT_LAYER)
        )

        content_loss = content_weight * (
            2 * tf.nn.l2_loss(_content_layer - content_layer) / content_size
        )

    # 2. Compute style loss.
    logger.info("Compute style reconstruction loss ratio.")

    with tf.name_scope("style_loss"):
        style_losses = []

        for layer_name, _ in stylish.vgg.STYLE_LAYERS:
            layer = session.graph.get_tensor_by_name(
                "loss_network/{}:0".format(layer_name)
            )

            shape = tf.shape(layer)
            new_shape = [shape[0], shape[1] * shape[2], shape[3]]
            tf_shape = tf.stack(new_shape)

            feats = tf.reshape(layer, shape=tf_shape)
            feats_transposed = tf.transpose(feats, perm=[0, 2, 1])

            size = tf.cast(shape[1] * shape[2] * shape[3], tf.float32)
            grams = tf.matmul(feats_transposed, feats) / size
            style_gram = style_features[layer_name]
            style_losses.append(
                2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size
            )

        style_loss = (style_weight * tf.reduce_sum(style_losses) / batch_size)

    # 3. Compute total variation loss.
    logger.info("Compute total variation loss ratio.")

    with tf.name_scope("tv_loss"):
        tv_y_size = tf.reduce_prod(
            tf.cast(tf.shape(input_node[:, 1:, :, :]), tf.float32)[1:]
        )
        tv_x_size = tf.reduce_prod(
            tf.cast(tf.shape(input_node[:, :, 1:, :]), tf.float32)[1:]
        )

        y_tv = tf.nn.l2_loss(
            input_node[:, 1:, :, :] - input_node[:, :batch_shape[1] - 1, :, :]
        )
        x_tv = tf.nn.l2_loss(
            input_node[:, :, 1:, :] - input_node[:, :, :batch_shape[2] - 1, :]
        )
        total_variation_loss = (
            tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size
        )

    return {
        "total": content_loss + style_loss + total_variation_loss,
        "content": content_loss,
        "style": style_loss,
        "total_variation": total_variation_loss
    }


def optimize(
    session, training_node, training_data, input_node, loss_mapping,
    output_log, output_checkpoint, batch_size=BATCH_SIZE,
    epoch_number=EPOCHS_NUMBER
):
    """Optimize the loss for *training_node*.

    *session* should be a :term:`Tensorflow` session.

    *training_node* should be the optimizer node that should be executed.

    *training_data* should be a list containing all training images to feed to
    the *input_node*.

    *input_node* should be the placeholder node in which should be feed each
    image from *training_data* to train the model.

    *loss_mapping* should be a mapping of all loss nodes as returned by
    :func:`compute_loss`.

    *output_log* should be the path to export the logs.

    *output_checkpoint* should be the path to export each checkpoints to
    resume the training at any time. A checkpoint will be saved after each
    epoch and at each 500 batches.

    *batch_size* should indicate the number of training examples utilized in one
    iteration. Default is :data:`BATCH_SIZE`.

    *epoch_number* should indicate the number of time that the *training data*
    should be trained. Default is :data:`EPOCHS_NUMBER`.

    """
    logger = stylish.logging.Logger(__name__ + ".optimize")

    # Initiate the batch shape.
    batch_shape = (batch_size, 256, 256, 3)

    # Initiate all variables.
    session.run(tf.global_variables_initializer())

    # Initiate the saver to export the checkpoints.
    saver = tf.train.Saver()

    # Save log to visualize the graph with tensorboard.
    train_writer = tf.summary.FileWriter(output_log, session.graph)
    total_cost = tf.summary.scalar(name="Total", tensor=loss_mapping["total"])
    content = tf.summary.scalar(name="Content", tensor=loss_mapping["content"])
    style = tf.summary.scalar(name="Style", tensor=loss_mapping["style"])
    total_variation = tf.summary.scalar(
        name="Total Variation", tensor=loss_mapping["total_variation"]
    )

    step = 0

    train_size = len(training_data)

    for epoch in range(epoch_number):
        logger.info("Start epoch #{}.".format(epoch))

        start_time_epoch = time.time()

        for index in range(train_size // batch_size):
            logger.debug("Start processing batch #{}.".format(index))
            start_time_batch = time.time()

            x_batch = get_next_batch(
                index, training_data, batch_size, batch_shape
            )

            # Execute the nodes within the session.
            _, _total_cost, _content, _style, _total_variation = session.run(
                [training_node, total_cost, content, style, total_variation],
                feed_dict={input_node: x_batch}
            )

            train_writer.add_summary(_total_cost, step)
            train_writer.add_summary(_content, step)
            train_writer.add_summary(_style, step)
            train_writer.add_summary(_total_variation, step)
            step += 1

            end_time_batch = time.time()

            message = (
                "Batch #{} processed [time: {}]"
                .format(index, end_time_batch - start_time_batch)
            )

            if index % 500 == 0:
                logger.info(message)
                saver.save(session, output_checkpoint)

            else:
                logger.debug(message)

        end_time_epoch = time.time()
        logger.info(
            "Epoch #{} processed [time: {}]"
            .format(epoch, end_time_epoch - start_time_epoch)
        )

        # Save checkpoint.
        saver.save(session, output_checkpoint)


def get_next_batch(iteration, content_targets, batch_size, batch_shape):
    """Return array with image matrices according to *iteration* index.

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
