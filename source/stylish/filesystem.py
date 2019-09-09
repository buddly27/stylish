# :coding: utf-8

import os
import re
import unicodedata
import errno

import numpy as np
import imageio
import skimage.transform

import stylish.logging


def load_image(path, image_size=None):
    """Return 3-D Numpy array from image *path*.

    :param path: path to image file to load.

    :param image_size: targeted size of the image matrix loaded. By default, the
        image will not be resized.

    :return: 3-D Numpy array representing the image loaded.

    .. note::

        `Lists of all formats currently supported
        <https://imageio.readthedocs.io/en/stable/formats.html>`_

    """
    logger = stylish.logging.Logger(__name__ + ".load_image")
    logger.debug("Load image from path: {!r}".format(path))

    image = imageio.imread(path).astype(np.float)

    # If the image is monochrome, make it into an RGB image.
    if not (len(image.shape) == 3 and image.shape[2] == 3):
        image = np.dstack((image, image, image))

    if image_size is not None:
        image = skimage.transform.resize(image, image_size)

    return image


def save_image(image, path):
    """Save *image_matrix* to *path*.

    :param image: 3-D Numpy array representing the image to save.

    :param path: path to image file to save *image* into.

    :return: None

    .. note::

        `Lists of all formats currently supported
        <https://imageio.readthedocs.io/en/stable/formats.html>`_

    """
    image = np.clip(image, 0, 255).astype(np.uint8)
    imageio.imwrite(path, image)


def fetch_images(path, limit=None):
    """Return list of image paths from *path*.

    :param path: path to the directory containing all images to fetch.

    :param limit: maximum number of files to fetch from *path*. By
    default, all files are fetched.

    :return: list of image file path.

    """
    if not os.path.isdir(path) or not os.access(path, os.R_OK):
        raise OSError("The image folder '{}' is incorrect".format(path))

    images = []

    for image in os.listdir(path)[:limit]:
        images.append(os.path.join(path, image))

    return images


def ensure_directory(path):
    """Ensure directory exists at *path*.

    :param path: directory path.

    :return: None

    """
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

    :param value: string value to sanitise.

    :param substitution_character: string character to replace awkward
        characters with. Default is "_".

    :param case_sensitive: indicate whether sanitised value should be kept with
        original case. Otherwise, the sanitised value will be return in
        lowercase. By default, the original case is preserved.

    :return: sanitised value.

    """
    value = unicodedata.normalize("NFKD", value)
    value = re.sub(r"[^\w._\-\\/:%]", substitution_character, value)
    value = value.strip()

    if not case_sensitive:
        value = value.lower()

    return value
