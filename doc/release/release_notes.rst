.. _release/release_notes:

*************
Release Notes
*************

.. release:: 0.1.3
    :date: 2018-05-19

    .. change:: fixed

        Reformat :mod:`stylish.train` module to prevent fixing the shape of the
        input placeholder.

.. release:: 0.1.2
    :date: 2018-05-18

    .. change:: fixed

        Reformat :mod:`stylish.transform` module to let the size of the images
        unknown when processing the checkpoint.

    .. change:: fixed

        Make the :func:`stylish.train.extract_model` function more verbose.

.. release:: 0.1.1
    :date: 2018-05-09

    .. change:: fixed

        Fixed :option:`--content-target <stylish train --content-target>`
        command line option as it should take a single value, not a list of
        values.

    .. change:: fixed

        Fixed :func:`stylish.train.extract_model` to pass the correct
        placeholder identifier to the session.

.. release:: 0.1.0
    :date: 2018-05-08

    .. change:: new

        Initial release.
