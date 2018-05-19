# :coding: utf-8

import os
import logging
import time

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

    style_name = os.path.basename(style_target.split(".", 1)[0])
    root = os.path.join(model_path, style_name)
    stylish.filesystem.ensure_directory_access(root)

    outputs = {
        "model": os.path.join(root, "model"),
        "checkpoints": os.path.join(root, "checkpoints"),
        "log": os.path.join(root, "log"),
    }

    # Pre-compute style feature map.
    style_features = compute_style_features(style_target, layers, mean_pixel)

    # Initiate a default graph.
    graph = tf.Graph()

    # Initiate the batch shape.
    batch_shape = (BATCH_SIZE, 256, 256, 3)

    with graph.as_default(), tf.Session() as session:
        input_placeholder = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name="input"
        )
        normalized_image = input_placeholder - mean_pixel

        with tf.name_scope("network"):
            loss_network = stylish.vgg.extract_network(normalized_image, layers)

        content_features = loss_network[stylish.vgg.CONTENT_LAYER]
        predictions = stylish.transform.network(input_placeholder/255.0)

        loss = compute_loss_ratio(
            predictions, batch_shape, style_features, content_features,
            layers, mean_pixel
        )

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        training_op = optimizer.minimize(loss)

        # Save log to visualize the graph with tensorboard.
        tf.summary.FileWriter(outputs["log"], session.graph)

        # Start training.

        logging.info("Start training.")

        session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        train_size = len(content_targets)

        for epoch in range(EPOCHS_NUMBER):
            logging.info("Start epoch #{}.".format(epoch))

            start_time = time.time()

            for index in range(train_size // BATCH_SIZE):
                logging.debug("Start processing batch #{}.".format(index))
                _start_time = time.time()

                x_batch = get_next_batch(
                    index, content_targets, BATCH_SIZE, batch_shape
                )

                session.run(training_op, feed_dict={input_placeholder: x_batch})

                _end_time = time.time()
                _delta = _end_time - _start_time

                message_batch_end = "Batch #{} processed [time: {}]."
                if index % 500 == 0:
                    logging.info(message_batch_end.format(index, _delta))
                    saver.save(session, outputs["checkpoints"])

                else:
                    logging.debug(message_batch_end.format(index, _delta))

            end_time = time.time()
            delta = end_time - start_time

            logging.info("Epoch #{} processed [time: {}].".format(epoch, delta))

            # Save checkpoint.

            saver.save(session, outputs["checkpoints"])

        # Save model.

        input_info = tf.saved_model.utils.build_tensor_info(input_placeholder)
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

    content_shape = tf.cast(tf.shape(content_features), tf.float32)
    content_size = tf.reduce_prod(content_shape[1:]) * BATCH_SIZE

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

        shape = tf.shape(layer)
        new_shape = [shape[0], shape[1] * shape[2], shape[3]]
        tf_shape = tf.stack(new_shape)

        feats = tf.reshape(layer, shape=tf_shape)
        feats_transposed = tf.transpose(feats, perm=[0, 2, 1])

        size = tf.cast(shape[1] * shape[2] * shape[3], tf.float32)
        grams = tf.matmul(feats_transposed, feats) / size
        style_gram = style_features[style_layer]
        style_losses.append(
            2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size
        )

    style_loss = (STYLE_WEIGHT * tf.reduce_sum(style_losses) / BATCH_SIZE)

    # Compute total variation de-noising.

    logging.info("Compute total variation loss ratio.")

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
