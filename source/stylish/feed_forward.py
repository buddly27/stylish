# :coding: utf-8

import os
import time
import logging

import tensorflow as tf
import numpy as np

import stylish.transform
import stylish.filesystem


def transform_image(model_file, input_image, output_path):
    """Transform input image with style generator model and return result.

    *model_file* should be the path to a Tensorflow checkpoint model file.

    *input_image* should be the path to an image to transform.

    *output_path* should be the folder where the output image should be saved.

    """
    logging.info("Apply style generator model.")

    # Extract image matrix from input image.
    image_matrix = stylish.filesystem.load_image(input_image)

    # Compute output image path.
    _input_image, _ = os.path.splitext(input_image)
    output_image = os.path.join(
        output_path, "{}.jpg".format(os.path.basename(_input_image))
    )

    # Initiate a default graph.
    graph = tf.Graph()

    # Initiate the batch shape.
    batch_shape = (1,) + image_matrix.shape

    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    with graph.as_default(), tf.Session(config=soft_config) as session:
        images = tf.placeholder(tf.float32, shape=batch_shape, name="images")
        transform = stylish.transform.network(images)

        logging.debug("Restore style generation model.")

        saver = tf.train.Saver()
        saver.restore(session, model_file)

        logging.debug("Start transformation.")

        start_time = time.time()

        predictions = session.run(
            transform, feed_dict={images: np.array([image_matrix])}
        )
        stylish.filesystem.save_image(predictions[0], output_image)

        end_time = time.time()
        logging.info(
            "Image transformed: {} [time={}]".format(
                output_image, end_time - start_time
            )
        )

        return output_image
