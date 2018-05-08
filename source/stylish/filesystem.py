# :coding: utf-8

import os
import errno
import logging

import numpy as np
import imageio
import scipy.misc


def load_image(image_path, image_size=None):
    """Return 3-D Numpy array from image *path*.

    *image_size* can be specified to resize the image.

    """
    logging.debug("Load image from path: {!r}".format(image_path))

    matrix = imageio.imread(image_path).astype(np.float)

    # If the image is monochrome, make it into an RGB image.
    if not (len(matrix.shape) == 3 and matrix.shape[2] == 3):
        matrix = np.dstack((matrix, matrix, matrix))

    if image_size is not None:
        matrix = scipy.misc.imresize(matrix, image_size)

    return matrix


def save_image(image_matrix, path):
    """Save *image_matrix* to *path*."""
    image = np.clip(image_matrix, 0, 255).astype(np.uint8)
    imageio.imwrite(path, image)


def fetch_images(path):
    """Return list of image paths from *path*."""
    if not os.path.isdir(path) or not os.access(path, os.R_OK):
        raise OSError("The image folder '{}' is incorrect".format(path))

    images = []

    for image in os.listdir(path):
        images.append(os.path.join(path, image))

    return images


def ensure_directory_access(path):
    """Ensure directory exists at *path* and is accessible."""
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise

        if not os.path.isdir(path):
            raise
