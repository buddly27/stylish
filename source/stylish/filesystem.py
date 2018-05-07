# :coding: utf-8

import numpy as np
import imageio
import logging


def load_image(image_path):
    """Return 3-D Numpy array from image *path*."""
    logging.debug("Load image from path: {!r}".format(image_path))

    matrix = imageio.imread(image_path).astype(np.float)

    # If the image is monochrome, make it into an RGB image.
    if not (len(matrix.shape) == 3 and matrix.shape[2] == 3):
        matrix = np.dstack((matrix, matrix, matrix))

    return matrix
