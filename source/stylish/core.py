# :coding: utf-8

import time
import contextlib
import datetime

import tensorflow as tf
import numpy as np

import stylish.logging
import stylish.filesystem
import stylish.vgg
import stylish.transform


#: Default batch size used for training.
BATCH_SIZE = 4

#: Default shape used for each images within training dataset.
BATCH_SHAPE = (256, 256, 3)

#: Default epoch number used for training a model.
EPOCHS_NUMBER = 2

#: Default iteration number used for transferring a style to an image.
ITERATIONS_NUMBER = 100

#: Default weight of the content for the loss computation.
CONTENT_WEIGHT = 7.5

#: Default weight of the style for the loss computation.
STYLE_WEIGHT = 100.0

#: Default weight of the total variation term for the loss computation.
TV_WEIGHT = 200.0

#: Default :term:`Learning Rate`.
LEARNING_RATE = 1e-3


@contextlib.contextmanager
def create_session():
    """Create a :term:`Tensorflow` session and reset the default graph.

    Should be used as follows::

        >>> with create_session() as session:
            ...

    :return: :term:`Tensorflow` session

    """
    tf.reset_default_graph()

    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    session = tf.Session(config=soft_config)

    try:
        yield session
    finally:
        session.close()


def extract_style_from_path(path, vgg_mapping, style_layers, image_size=None):
    """Extract style feature mapping from image *path*.

    This mapping will be used to train a model which should learn to apply those
    features on any images.

    :param path: path to image from which style features will be extracted.

    :param vgg_mapping: mapping gathering all weight and bias matrices extracted
        from a pre-trained :term:`Vgg19` model (typically retrieved by
        :func:`stylish.vgg.extract_mapping`).

    :param style_layers: Layer names from pre-trained :term:`Vgg19` model
        used to extract the style information with corresponding weights.
        Default is :data:`stylish.vgg.STYLE_LAYERS`.

    :param image_size: optional shape to resize the style image.

    list of 5 values for each layer used for
    style features extraction. Default is :data:`LAYER_WEIGHTS`.

    :return:
        mapping in the form of::

            {
                "conv1_1/Relu": numpy.array([...]),
                "conv2_1/Relu": numpy.array([...]),
                "conv3_1/Relu": numpy.array([...]),
                "conv4_1/Relu": numpy.array([...]),
                "conv5_1/Relu": numpy.array([...])
            }

    """
    logger = stylish.logging.Logger(__name__ + ".extract_style_from_path")

    # Load image from path.
    image = stylish.filesystem.load_image(path, image_size=image_size)

    # Initiate the shape of a 4-D Tensor for a list of images.
    image_shape = (1,) + image.shape

    # Initiate style feature mapping.
    mapping = {}

    with create_session() as session:
        input_node = tf.placeholder(tf.float32, shape=image_shape, name="input")
        input_node = input_node - stylish.vgg.VGG19_MEAN

        with tf.name_scope("vgg"):
            stylish.vgg.network(vgg_mapping, input_node)

        # Initiate input as a list of images.
        images = np.array([image])

        # Get default graph augmented with VGG network.
        graph = tf.get_default_graph()

        for layer_name, weight in style_layers:
            logger.info(
                "Extracting features from layer '{}' [weight: {}]".format(
                    layer_name, weight
                )
            )

            layer = graph.get_tensor_by_name("vgg/{}:0".format(layer_name))

            # Run session on style layer.
            features = session.run(layer, feed_dict={input_node: images})
            logger.debug("Layer '{}' processed.".format(layer_name))

            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            mapping[layer_name] = gram * weight

        return mapping


def optimize_image(
    image, style_mapping, vgg_mapping, log_path, iterations=None,
    learning_rate=None, content_weight=None, style_weight=None, tv_weight=None,
    content_layer=None, style_layer_names=None
):
    """Transfer style mapping features to *image* and return result.

    The training duration can vary depending on the :term:`Hyperparameters
    <Hyperparameter>` specified (iterations number) and the power
    of your workstation.

    :param image: 3-D Numpy array representing the image loaded.

    :param style_mapping: mapping of pre-computed style features extracted from
        selected layers from a pre-trained :term:`Vgg19` model (typically
        retrieved by :func:`extract_style_from_path`)

    :param vgg_mapping: mapping gathering all weight and bias matrices extracted
        from a pre-trained :term:`Vgg19` model (typically retrieved by
        :func:`stylish.vgg.extract_mapping`).

    :param log_path: path to save the log information into, so it can be used
        with :term:`Tensorboard` to analyze the training.

    :param iterations: number of time that image should be trained against
        the style mapping. Default is :data:`ITERATIONS_NUMBER`.

    :param learning_rate: :term:`Learning Rate` value to train the model.
        Default is :data:`LEARNING_RATE`.

    :param content_weight: weight of the content feature cost. Default is
        :data:`CONTENT_WEIGHT`.

    :param style_weight: weight of the style feature cost. Default is
        :data:`STYLE_WEIGHT`.

    :param tv_weight: weight of the total variation cost. Default is
        :data:`TV_WEIGHT`.

    :param content_layer: Layer name from pre-trained :term:`Vgg19` model
        used to extract the content information. Default is
        :data:`stylish.vgg.CONTENT_LAYER`.

    :param style_layer_names: Layer names from pre-trained :term:`Vgg19` model
        used to extract the style information. Default are layer names extracted
        from :data:`stylish.vgg.STYLE_LAYERS` tuples.

    :return: Path to output image generated.

    """
    logger = stylish.logging.Logger(__name__ + ".optimize_image")

    with create_session() as session:
        input_node = tf.placeholder(
            tf.float32, shape=(1,) + image.shape, name="input"
        )

        # Build main network.
        output_node = stylish.transform.network(
            (input_node - stylish.vgg.VGG19_MEAN) / 255.0
        )

        # Save image.
        tf.summary.image("image", tensor=output_node)

        # Build loss networks.
        with tf.name_scope("vgg1"):
            stylish.vgg.network(
                vgg_mapping, input_node - stylish.vgg.VGG19_MEAN
            )

        with tf.name_scope("vgg2"):
            stylish.vgg.network(
                vgg_mapping, output_node - stylish.vgg.VGG19_MEAN
            )

        # Compute total cost.
        cost = compute_cost(
            session, style_mapping, output_node,
            batch_size=1,
            content_weight=content_weight or CONTENT_WEIGHT,
            style_weight=style_weight or STYLE_WEIGHT,
            tv_weight=tv_weight or TV_WEIGHT,
            content_layer=content_layer or stylish.vgg.CONTENT_LAYER,
            style_layer_names=style_layer_names or [
                name for name, _ in stylish.vgg.STYLE_LAYERS
            ],
            input_namespace="vgg1",
            output_namespace="vgg2"
        )

        # Apply optimizer to attempt to reduce the total cost.
        optimizer = tf.train.AdamOptimizer(learning_rate or LEARNING_RATE)
        training_node = optimizer.minimize(cost)

        # Add graph to writer to visualize it with tensorboard.
        writer = tf.summary.FileWriter(log_path, graph=session.graph)

        # Initiate all variables.
        session.run(tf.global_variables_initializer())

        # Merges all summaries collected in the default graph.
        merged_summary = tf.summary.merge_all()

        # Initiate timer and predictions.
        start_time = time.time()

        for iteration in range(iterations or ITERATIONS_NUMBER):
            logger.debug("Start processing iteration #{}.".format(iteration))
            start_time_iteration = time.time()

            # Execute the nodes within the session.
            _, summary = session.run(
                [training_node, merged_summary],
                feed_dict={input_node: np.array([image])}
            )
            writer.add_summary(summary, iteration)

            end_time_iteration = time.time()
            duration = end_time_iteration - start_time_iteration

            logger.info(
                "Iteration {}/{} processed [duration: {} - total: {}]"
                .format(
                    iteration, iterations or ITERATIONS_NUMBER,
                    datetime.timedelta(seconds=duration),
                    datetime.timedelta(seconds=end_time_iteration - start_time)
                )
            )

        images = session.run(
            output_node, feed_dict={input_node: np.array([image])}
        )
        return images[0]


def optimize_model(
    training_images, style_mapping, vgg_mapping, model_path, log_path,
    learning_rate=None, batch_size=None, batch_shape=None, epoch_number=None,
    content_weight=None, style_weight=None, tv_weight=None, content_layer=None,
    style_layer_names=None
):
    """Create style generator model from a style mapping and a training dataset.

    The training duration can vary depending on the :term:`Hyperparameters
    <Hyperparameter>` specified (epoch number, batch size, etc.), the power
    of your workstation and the number of images in the training data.

    The model trained will be saved in *model_path*.

    :param training_images: list of images to train the model with.

    :param style_mapping: mapping of pre-computed style features extracted from
        selected layers from a pre-trained :term:`Vgg19` model (typically
        retrieved by :func:`extract_style_from_path`)

    :param vgg_mapping: mapping gathering all weight and bias matrices extracted
        from a pre-trained :term:`Vgg19` model (typically retrieved by
        :func:`stylish.vgg.extract_mapping`).

    :param model_path: path to save the trained model into.

    :param log_path: path to save the log information into, so it can be used
        with :term:`Tensorboard` to analyze the training.

    :param learning_rate: :term:`Learning Rate` value to train the model.
        Default is :data:`LEARNING_RATE`.

    :param batch_size: number of images to use in one training iteration.
        Default is :data:`BATCH_SIZE`.

    :param batch_shape: shape used for each images within training dataset.
        Default is :data:`BATCH_SHAPE`.

    :param epoch_number: number of time that model should be trained against
        *training_images*. Default is :data:`EPOCHS_NUMBER`.

    :param content_weight: weight of the content feature cost. Default is
        :data:`CONTENT_WEIGHT`.

    :param style_weight: weight of the style feature cost. Default is
        :data:`STYLE_WEIGHT`.

    :param tv_weight: weight of the total variation cost. Default is
        :data:`TV_WEIGHT`.

    :param content_layer: Layer name from pre-trained :term:`Vgg19` model
        used to extract the content information. Default is
        :data:`stylish.vgg.CONTENT_LAYER`.

    :param style_layer_names: Layer names from pre-trained :term:`Vgg19` model
        used to extract the style information. Default are layer names extracted
        from :data:`stylish.vgg.STYLE_LAYERS` tuples.

    :return: None

    """
    logger = stylish.logging.Logger(__name__ + ".optimize_model")

    with create_session() as session:
        input_node = tf.placeholder(
            tf.float32, shape=(None, None, None, None), name="input"
        )

        # Build main network.
        output_node = stylish.transform.network(
            (input_node - stylish.vgg.VGG19_MEAN) / 255.0
        )

        # Add dummy output node that can be targeted for model application
        output_node = tf.identity(output_node, name="output")

        # Train the network on training data
        # Build loss networks.
        with tf.name_scope("vgg1"):
            stylish.vgg.network(
                vgg_mapping, input_node - stylish.vgg.VGG19_MEAN
            )

        with tf.name_scope("vgg2"):
            stylish.vgg.network(
                vgg_mapping, output_node - stylish.vgg.VGG19_MEAN
            )

        # Compute total cost.
        cost = compute_cost(
            session, style_mapping, output_node,
            batch_size=batch_size or BATCH_SIZE,
            content_weight=content_weight or CONTENT_WEIGHT,
            style_weight=style_weight or STYLE_WEIGHT,
            tv_weight=tv_weight or TV_WEIGHT,
            content_layer=content_layer or stylish.vgg.CONTENT_LAYER,
            style_layer_names=style_layer_names or [
                name for name, _ in stylish.vgg.STYLE_LAYERS
            ],
            input_namespace="vgg1",
            output_namespace="vgg2"
        )

        # Apply optimizer to attempt to reduce the total cost.
        optimizer = tf.train.AdamOptimizer(learning_rate or LEARNING_RATE)
        training_node = optimizer.minimize(cost)

        # Add graph to writer to visualize it with tensorboard.
        writer = tf.summary.FileWriter(log_path, graph=session.graph)

        # Initiate all variables.
        session.run(tf.global_variables_initializer())

        # Merges all summaries collected in the default graph.
        merged_summary = tf.summary.merge_all()

        iteration = 0
        start_time = time.time()

        train_size = len(training_images)

        for epoch in range(epoch_number or EPOCHS_NUMBER):
            logger.info("Start epoch #{}.".format(epoch))

            start_time_epoch = time.time()

            for index in range(train_size // (batch_size or BATCH_SIZE)):
                logger.debug("Start processing batch #{}.".format(index))
                start_time_batch = time.time()

                images = load_dataset_batch(
                    index, training_images,
                    batch_size=batch_size or BATCH_SIZE,
                    batch_shape=batch_shape or BATCH_SHAPE
                )

                # Execute the nodes within the session.
                _, summary = session.run(
                    [training_node, merged_summary],
                    feed_dict={input_node: images}
                )
                writer.add_summary(summary, iteration)
                iteration += 1

                end_time_batch = time.time()
                batch_duration = end_time_batch - start_time_batch

                message = (
                    "Batch #{} processed [duration: {} - total: {}]"
                    .format(
                        index,
                        datetime.timedelta(seconds=batch_duration),
                        datetime.timedelta(seconds=end_time_batch - start_time)
                    )
                )

                if index % 500 == 0:
                    logger.info(message)

                else:
                    logger.debug(message)

            end_time_epoch = time.time()
            epoch_duration = end_time_epoch - start_time_epoch
            logger.info(
                "Epoch #{} processed [duration: {} - total: {}]"
                .format(
                    epoch,
                    datetime.timedelta(seconds=epoch_duration),
                    datetime.timedelta(seconds=end_time_epoch - start_time)
                )
            )

        # Save model.
        save_model(session, input_node, output_node, model_path)


def compute_cost(
    session, style_mapping, output_node, batch_size=None, content_weight=None,
    style_weight=None, tv_weight=None, content_layer=None,
    style_layer_names=None, input_namespace="vgg1", output_namespace="vgg2"
):
    """Compute total cost.

    :param session: :term:`Tensorflow` session.

    :param style_mapping: mapping of pre-computed style features extracted from
        selected layers from a pre-trained :term:`Vgg19` model (typically
        retrieved by :func:`extract_style_from_path`)

    :param output_node: output node of the model to train.

    :param batch_size: number of images to use in one training iteration.
        Default is :data:`BATCH_SIZE`.

    :param content_weight: weight of the content feature cost. Default is
        :data:`CONTENT_WEIGHT`.

    :param style_weight: weight of the style feature cost. Default is
        :data:`STYLE_WEIGHT`.

    :param tv_weight: weight of the total variation cost. Default is
        :data:`TV_WEIGHT`.

    :param content_layer: Layer name from pre-trained :term:`Vgg19` model
        used to extract the content information. Default is
        :data:`stylish.vgg.CONTENT_LAYER`.

    :param style_layer_names: Layer names from pre-trained :term:`Vgg19` model
        used to extract the style information. Default are layer names extracted
        from :data:`stylish.vgg.STYLE_LAYERS` tuples.

    :param input_namespace: Namespace used for the pre-trained :term:`Vgg19`
        model added after the input node. Default is "vgg1".

    :param output_namespace: Namespace used for the pre-trained :term:`Vgg19`
        model added after *output_node*. Default is "vgg2".

    :return: Tensor computing the total cost.

    """
    content_layer = content_layer or stylish.vgg.CONTENT_LAYER
    style_layer_names = style_layer_names or [
        name for name, _ in stylish.vgg.STYLE_LAYERS
    ]

    # Compute content cost.
    content_cost = compute_content_cost(
        session,
        "{}/{}:0".format(input_namespace, content_layer),
        "{}/{}:0".format(output_namespace, content_layer),
        batch_size=batch_size or BATCH_SIZE,
        content_weight=content_weight or CONTENT_WEIGHT
    )

    # Compute style cost.
    style_cost = compute_style_cost(
        session, style_mapping,
        style_layer_names,
        [
            "{}/{}:0".format(output_namespace, name)
            for name in style_layer_names
        ],
        batch_size=batch_size or BATCH_SIZE,
        style_weight=style_weight or STYLE_WEIGHT
    )

    # Compute total variation cost.
    total_variation_cost = compute_total_variation_cost(
        output_node, batch_size or BATCH_SIZE,
        tv_weight=tv_weight or TV_WEIGHT
    )

    cost = content_cost + style_cost + total_variation_cost
    tf.summary.scalar("total", tensor=cost)
    return cost


def compute_content_cost(
    session, layer_name1, layer_name2, batch_size=BATCH_SIZE,
    content_weight=CONTENT_WEIGHT
):
    """Compute content cost.

    :param session: :term:`Tensorflow` session.

    :param layer_name1: Layer name from pre-trained :term:`Vgg19` model
        used to extract the content information of input node.

    :param layer_name2: Layer name from pre-trained :term:`Vgg19` model
        used to extract the content information of output node.

    :param batch_size: number of images to use in one training iteration.
        Default is :data:`BATCH_SIZE`.

    :param content_weight: weight of the content feature cost. Default is
        :data:`CONTENT_WEIGHT`.

    :return: Tensor computing the content cost.

    """
    layer1 = session.graph.get_tensor_by_name(layer_name1)

    with tf.name_scope("content_loss"):
        content_shape = tf.cast(tf.shape(layer1), tf.float32)
        content_size = tf.reduce_prod(content_shape[1:]) * batch_size
        layer2 = session.graph.get_tensor_by_name(layer_name2)

        cost = 2 * tf.nn.l2_loss(layer2 - layer1) / content_size
        cost = content_weight * cost

    tf.summary.scalar("content", tensor=cost)
    return cost


def compute_style_cost(
    session, style_mapping, layer_names1, layer_names2, batch_size=BATCH_SIZE,
    style_weight=STYLE_WEIGHT
):
    """Compute style cost.

    :param session: :term:`Tensorflow` session.

    :param style_mapping: mapping of pre-computed style features extracted from
        selected layers from a pre-trained :term:`Vgg19` model (typically
        retrieved by :func:`extract_style_from_path`)

    :param layer_names1: Sorted layer names used in *style_mapping*.

    :param layer_names2: Layer name from pre-trained :term:`Vgg19` model
        used to extract the style information of output node.

    :param batch_size: number of images to use in one training iteration.
        Default is :data:`BATCH_SIZE`.

    :param style_weight: weight of the style feature cost. Default is
        :data:`STYLE_WEIGHT`.

    :return: Tensor computing the style cost.

    """
    with tf.name_scope("style_loss"):
        style_losses = []

        for layer_name1, layer_name2 in zip(layer_names1, layer_names2):
            layer = session.graph.get_tensor_by_name(layer_name2)

            shape = tf.shape(layer)
            new_shape = [shape[0], shape[1] * shape[2], shape[3]]
            tf_shape = tf.stack(new_shape)

            features = tf.reshape(layer, shape=tf_shape)
            features_transposed = tf.transpose(features, perm=[0, 2, 1])

            style_size = tf.cast(shape[1] * shape[2] * shape[3], tf.float32)
            grams = tf.matmul(features_transposed, features) / style_size

            style_gram = style_mapping[layer_name1]
            style_losses.append(
                2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size
            )

        cost = tf.reduce_sum(style_losses) / batch_size
        cost = style_weight * cost

    tf.summary.scalar("style", tensor=cost)
    return cost


def compute_total_variation_cost(output_node, batch_size, tv_weight=TV_WEIGHT):
    """Compute total variation cost.

    :param output_node: output node of the model to train.

    :param batch_size: number of images to use in one training iteration.

    :param tv_weight: weight of the total variation cost. Default is
        :data:`TV_WEIGHT`.

    :return: Tensor computing the total variation cost.

    """
    with tf.name_scope("tv_loss"):
        tv_y_size = tf.reduce_prod(
            tf.cast(tf.shape(output_node[:, 1:, :, :]), tf.float32)[1:]
        )
        tv_x_size = tf.reduce_prod(
            tf.cast(tf.shape(output_node[:, :, 1:, :]), tf.float32)[1:]
        )

        y_tv = tf.nn.l2_loss(
            output_node[:, 1:, :, :] - output_node[:, :-1, :, :]
        )
        x_tv = tf.nn.l2_loss(
            output_node[:, :, 1:, :] - output_node[:, :, :-1, :]
        )

        cost = 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size
        cost = tv_weight * cost

    tf.summary.scalar("total_variation", tensor=cost)
    return cost


def load_dataset_batch(
    index, training_images, batch_size=None, batch_shape=None
):
    """Return list of images for current batch *index*.

    Usage example::

        >>> for index in range(len(training_images) // batch_size)):
        ...     images = load_dataset_batch(
        ...         index, training_images,
        ...         batch_size=batch_size
        ...     )

    :param index: index number of the current batch to load.

    :param training_images: complete list of images to train the model with.

    :param batch_size: number of images to use in one training iteration.
        Default is :data:`BATCH_SIZE`.

    :param batch_shape: shape used for each images within training dataset.
        Default is :data:`BATCH_SHAPE`.

    :return: 4-dimensional matrix storing images in batch.

    """
    batch_size = batch_size or BATCH_SIZE
    batch_shape = batch_shape or BATCH_SHAPE

    current = index * batch_size
    step = current + batch_size

    images = np.zeros((batch_size,) + batch_shape, dtype=np.float32)

    # Extract and resize images from training data.
    for index, image_path in enumerate(training_images[current:step]):
        images[index] = stylish.filesystem.load_image(
            image_path, image_size=batch_shape
        )

    return images


def save_model(session, input_node, output_node, path):
    """Save trained model from *session*.

    :param session: :term:`Tensorflow` session.

    :param input_node: input placeholder node of the model trained.

    :param output_node: output node of the model trained.

    :param path: Path to save the model into.

    :return: None

    """
    input_info = tf.compat.v1.saved_model.build_tensor_info(input_node)
    output_info = tf.compat.v1.saved_model.build_tensor_info(output_node)

    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"input": input_info},
        outputs={"output": output_info},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder = tf.saved_model.builder.SavedModelBuilder(path)
    builder.add_meta_graph_and_variables(
        session, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={"predict_images": signature},

    )
    builder.save()


def infer_model(model_path, input_path):
    """Inferred trained model to convert input image.

    :param model_path: path to trained model saved.

    :param input_path: path to image to inferred model to.

    :return: Path to output image generated.

    """
    logger = stylish.logging.Logger(__name__ + ".infer_model")

    # Extract image matrix from input image.
    image = stylish.filesystem.load_image(input_path)

    with create_session() as session:
        graph = tf.get_default_graph()

        tf.compat.v1.saved_model.load(session, ["serve"], model_path)
        input_node = graph.get_tensor_by_name("input:0")
        output_node = graph.get_tensor_by_name("output:0")

        start_time = time.time()

        images = session.run(
            output_node, feed_dict={input_node: np.array([image])}
        )

        end_time = time.time()
        logger.info(
            "Inferred model [duration: {}]".format(
                datetime.timedelta(seconds=end_time - start_time)
            )
        )

        return images[0]
