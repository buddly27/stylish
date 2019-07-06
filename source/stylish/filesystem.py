# :coding: utf-8

import os
import re
import unicodedata
import errno

import numpy as np
import imageio
import skimage.transform

import stylish.logging


def load_image(image_path, image_size=None):
    """Return 3-D Numpy array from image *path*.

    *image_size* can be specified to resize the image.

    """
    logger = stylish.logging.Logger(__name__ + ".load_image")
    logger.debug("Load image from path: {!r}".format(image_path))

    matrix = imageio.imread(image_path).astype(np.float)

    # If the image is monochrome, make it into an RGB image.
    if not (len(matrix.shape) == 3 and matrix.shape[2] == 3):
        matrix = np.dstack((matrix, matrix, matrix))

    if image_size is not None:
        matrix = skimage.transform.resize(matrix, image_size)

    return matrix


def save_image(image_matrix, path):
    """Save *image_matrix* to *path*."""
    image = np.clip(image_matrix, 0, 255).astype(np.uint8)
    imageio.imwrite(path, image)


def fetch_images(path, limit=None):
    """Return list of image paths from *path*.

    *limit* should be the maximum number of files to fetch from *path*. By
    default, all files are fetched.

    """
    if not os.path.isdir(path) or not os.access(path, os.R_OK):
        raise OSError("The image folder '{}' is incorrect".format(path))

    images = []

    for image in os.listdir(path)[:limit]:
        images.append(os.path.join(path, image))

    return images


def ensure_directory(path):
    """Ensure directory exists at *path*."""
    # Explicitly indicate that path should be a directory as default OSError
    # raised by 'os.makedirs' just indicates that the file exists, which is a
    # bit confusing for user.
    if os.path.isfile(path):
        raise OSError("'{}' should be a directory".format(path))

    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise

        if not os.path.isdir(path):
            raise


def sanitise_value(value, substitution_character="_", case_sensitive=True):
    """Return *value* suitable for use with filesystem.

    Replace awkward characters with *substitution_character*. Where possible,
    convert unicode characters to their closest "normal" form.

    If not *case_sensitive*, then also lowercase value.

    """
    value = unicodedata.normalize("NFKD", value)
    value = re.sub(r"[^\w._\-\\/:%]", substitution_character, value)
    value = value.strip()

    if not case_sensitive:
        value = value.lower()

    return value
