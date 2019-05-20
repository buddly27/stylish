# :coding: utf-8

import os
import time

import tensorflow as tf
import numpy as np

import stylish.logging
import stylish.filesystem
import stylish.model
import stylish.transform
from stylish._version import __version__


BATCH_SIZE = 4
CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2
LEARNING_RATE = 1e-3
EPOCHS_NUMBER = 2


def train_model(
    style_image_path, content_directory, output_directory, vgg_model_path
):
    """Train and return style generator model path.

    *style_image_path* should be the path to an image from which the style
    features will be extracted.

    *content_directory* should be a folder containing images from which the
    content features will be extracted.

    *output_directory* should be the path where the trained model should be
    saved.

    *vgg_model_path* should be the path to the Vgg19 pre-trained model in the
    MatConvNet data format.

    """
    logger = stylish.logging.Logger(__name__ + ".train_model")
    logger.info("Train model for style image: {}".format(style_image_path))

    # Extract weight and bias from pre-trained Vgg19 mapping.
    vgg_mapping = stylish.model.extract_mapping(vgg_model_path)

    # # Extract targeted images for training.
    logger.info("Extract content images from '{}'".format(content_directory))
    content_paths = stylish.filesystem.fetch_images(content_directory)
    logger.info("{} content image(s) found.".format(len(content_paths)))

    # Build the folder output.
    style_name = os.path.basename(style_image_path.split(".", 1)[0])
    root = os.path.join(
        output_directory,
        stylish.filesystem.sanitise_value(style_name, case_sensitive=False)
    )
    stylish.filesystem.ensure_directory(root)
    logger.info("Output model and logs will be in: {}".format(root))

    outputs = {
        "model": os.path.join(root, "model"),
        "checkpoints": os.path.join(root, "checkpoints"),
        "log": os.path.join(root, "log"),
    }

    # Pre-compute style features.
    style_features = compute_style_features(style_image_path, vgg_mapping)

    # Initiate a default graph.
    graph = tf.Graph()

    # Initiate the batch shape.
    batch_shape = (BATCH_SIZE, 256, 256, 3)

    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    with graph.as_default(), tf.Session(config=soft_config) as session:
        content_input = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name="input"
        )

        normalized_image = content_input - stylish.model.VGG19_MEAN

        with tf.name_scope("network1"):
            stylish.model.loss_network(vgg_mapping, normalized_image)

        content_layer = graph.get_tensor_by_name(
            "network1/{}:0".format(stylish.model.CONTENT_LAYER)
        )
        predictions = stylish.transform.network(content_input/255.0)

        loss = compute_loss_ratio(
            predictions, batch_shape, style_features, content_layer, vgg_mapping
        )

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        training_op = optimizer.minimize(loss)

        # Save log to visualize the graph with tensorboard.
        tf.summary.FileWriter(outputs["log"], session.graph)

        # Start training.
        logger.info("Start training.")

        session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        train_size = len(content_paths)

        for epoch in range(EPOCHS_NUMBER):
            logger.info("Start epoch #{}.".format(epoch))

            start_time = time.time()

            for index in range(train_size // BATCH_SIZE):
                logger.debug("Start processing batch #{}.".format(index))
                _start_time = time.time()

                x_batch = get_next_batch(
                    index, content_paths, BATCH_SIZE, batch_shape
                )

                session.run(training_op, feed_dict={content_input: x_batch})

                _end_time = time.time()
                _delta = _end_time - _start_time

                message_batch_end = "Batch #{} processed [time: {}]."
                if index % 500 == 0:
                    logger.info(message_batch_end.format(index, _delta))
                    saver.save(session, outputs["checkpoints"])

                else:
                    logger.debug(message_batch_end.format(index, _delta))

            end_time = time.time()
            delta = end_time - start_time

            logger.info("Epoch #{} processed [time: {}].".format(epoch, delta))

            # Save checkpoint.

            saver.save(session, outputs["checkpoints"])

        # Save model.

        input_info = tf.saved_model.utils.build_tensor_info(content_input)
        output_info = tf.saved_model.utils.build_tensor_info(predictions)

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"input": input_info},
            outputs={"output": output_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        builder = tf.saved_model.builder.SavedModelBuilder(outputs["model"])
        builder.add_meta_graph_and_variables(
            session, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={"predict_images": signature},

        )
        builder.save()

        return outputs["model"]


def apply_model(model_file, input_image, output_path):
    """Transform input image with style generator model and return result.

    *model_file* should be the path to a :term:`Tensorflow` checkpoint model
    file path.

    *input_image* should be the path to an image to transform.

    *output_path* should be the folder where the output image should be saved.

    """
    logger = stylish.logging.Logger(__name__ + ".apply_model")
    logger.info("Apply style generator model.")

    # Extract image matrix from input image.
    image_matrix = stylish.filesystem.load_image(input_image)

    # Compute output image path.
    _input_image, _ = os.path.splitext(input_image)
    output_image = os.path.join(
        output_path, "{}.jpg".format(os.path.basename(_input_image))
    )

    # Initiate a default graph.
    graph = tf.Graph()

    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    with graph.as_default(), tf.Session(config=soft_config) as session:
        tf.saved_model.loader.load(session, ["serve"], model_file)
        output_tensor = graph.get_tensor_by_name("output:0")
        input_tensor = graph.get_tensor_by_name("input:0")

        start_time = time.time()

        predictions = session.run(
            output_tensor, feed_dict={input_tensor: np.array([image_matrix])}
        )
        stylish.filesystem.save_image(predictions[0], output_image)

        end_time = time.time()
        logger.info(
            "Image transformed: {} [time={}]".format(
                output_image, end_time - start_time
            )
        )

        return output_image


def compute_style_features(path, vgg_mapping):
    """Return computed style features map from *style_path*.

    The style feature map will be used to penalize the predicted image when it
    deviates from the style (colors, textures, common patterns, etc.).

    *vgg_mapping* should gather all weight and bias matrices extracted from a
    pre-trained Vgg19 model (e.g. :func:`extract_mapping`).

    """
    logger = stylish.logging.Logger(__name__ + ".compute_style_features")
    logger.info("Extract style features from path: {}".format(path))

    # Extract image matrix from image.
    image_matrix = stylish.filesystem.load_image(path)

    # Initiate a default graph.
    graph = tf.Graph()

    # Initiate the shape of a 4-D Tensor for a list of images.
    image_shape = (1,) + image_matrix.shape

    # Initiate the style features.
    style_features = {}

    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    with graph.as_default(), tf.Session(config=soft_config) as session:
        style_input = tf.placeholder(
            tf.float32, shape=image_shape, name="style_input"
        )
        normalized_image = style_input - stylish.model.VGG19_MEAN

        with tf.name_scope("network2"):
            stylish.model.loss_network(vgg_mapping, normalized_image)

        # Initiate input as a list of images.
        images = np.array([image_matrix])

        for layer_name, weight in stylish.model.STYLE_LAYERS:
            logger.debug("Processing style layer '{}'".format(layer_name))
            layer = graph.get_tensor_by_name("network2/{}:0".format(layer_name))

            features = session.run(layer, feed_dict={style_input: images})
            logger.debug("Layer '{}' processed.".format(layer_name))

            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer_name] = gram * weight

    return style_features


def compute_loss_ratio(
    predictions, batch_shape, style_features, content_layer, vgg_mapping
):
    """Compute loss ratio from *predictions*.

    *predictions* should be the output tensor of the
    :func:`transformation network <stylish.transform.network>`.

    *batch_shape* should be the shape of the 4-D input image list tensor.

    *style_features* should be the style features map :func:`extracted
    <compute_style_features>`.

    *content_layer* should be the layer chosen to analyze the content features.

    *vgg_mapping* should gather all weight and bias matrices extracted from a
    pre-trained Vgg19 model (e.g. :func:`extract_mapping`).

    """
    logger = stylish.logging.Logger(__name__ + ".compute_loss_ratio")

    normalized_predictions = predictions - stylish.model.VGG19_MEAN

    with tf.name_scope("network3"):
        stylish.model.loss_network(vgg_mapping, normalized_predictions)

    # Compute feature reconstruction loss from content feature map.

    logger.info("Compute feature reconstruction loss ratio.")

    content_shape = tf.cast(tf.shape(content_layer), tf.float32)
    content_size = tf.reduce_prod(content_shape[1:]) * BATCH_SIZE
    _content_layer = tf.get_default_graph().get_tensor_by_name(
        "network3/{}:0".format(stylish.model.CONTENT_LAYER)
    )

    content_loss = CONTENT_WEIGHT * (
        2 * tf.nn.l2_loss(_content_layer - content_layer) / content_size
    )

    # Compute style reconstruction loss from style features map.

    logger.info("Compute style reconstruction loss ratio.")

    style_losses = []

    for layer_name, _ in stylish.model.STYLE_LAYERS:
        layer = tf.get_default_graph().get_tensor_by_name(
            "network3/{}:0".format(layer_name)
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

    style_loss = (STYLE_WEIGHT * tf.reduce_sum(style_losses) / BATCH_SIZE)

    # Compute total variation de-noising.

    logger.info("Compute total variation loss ratio.")

    tv_y_size = tf.reduce_prod(
        tf.cast(tf.shape(predictions[:, 1:, :, :]), tf.float32)[1:]
    )
    tv_x_size = tf.reduce_prod(
        tf.cast(tf.shape(predictions[:, :, 1:, :]), tf.float32)[1:]
    )

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
