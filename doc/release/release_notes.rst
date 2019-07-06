.. _release/release_notes:

*************
Release Notes
*************

.. changelog::
    :version: 0.3.0
    :released: 2019-07-05

    .. change:: new

        Added ``stylish train --limit`` option to set a maximum number of files
        to use from the training dataset folder.

    .. change:: new

        Record style loss, content loss, total variation loss and the sum of all
        losses to generate scalar curves within Tensorboard.

.. changelog::
    :version: 0.2.0
    :released: 2019-05-27

    .. change:: new

        Added ``stylish download`` command line option to download elements
        necessary for the training (:term:`Vgg19` model and training data).

    .. change:: new

        Added :mod:`stylish.logging` to manage logger using `sawmill
        <https://sawmill.readthedocs.io/en/latest/>`_ for convenience.

    .. change:: changed

        Removed :mod:`stylish.train` and moved logic within :mod:`stylish` to
        increase code readability.

    .. change:: fixed
        :tags: command line

        Updated :mod:`stylish.command_line` to use
        `click <https://pypi.org/project/click/>`_ instead of `argparse
        <https://docs.python.org/3/library/argparse.html>`_ to manage the
        command line interface for convenience.

    .. change:: fixed

        Fixed :func:`stylish.transform.instance_normalization` logic.

.. changelog::
    :version: 0.1.4
    :released: 2018-05-19

    .. change:: new

        Always use GPU for the training when available.

.. changelog::
    :version: 0.1.3
    :released: 2018-05-19

    .. change:: fixed

        Updated :mod:`stylish.train` module to prevent fixing the shape of the
        input placeholder.

.. changelog::
    :version: 0.1.2
    :released: 2018-05-18

    .. change:: fixed

        Updated :mod:`stylish.transform` module to let the size of the images
        unknown when processing the checkpoint.

    .. change:: fixed

        Updated :func:`stylish.train.extract_model` to increase verbosity.

.. changelog::
    :version: 0.1.1
    :released: 2018-05-09

    .. change:: fixed

        Fixed ``--content-target`` command line option as it should take a
        single value, not a list of values.

    .. change:: fixed

        Fixed :func:`stylish.train.extract_model` to pass the correct
        placeholder identifier to the session.

.. changelog::
    :version: 0.1.0
    :released: 2018-05-08

    .. change:: new

        Initial release.
