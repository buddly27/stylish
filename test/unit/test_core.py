# :coding: utf-8

import pytest
import tensorflow as tf
import numpy as np

import stylish.core


@pytest.fixture()
def mocked_tf_session(mocker):
    """Return mocked :func:`tensorflow.Session`."""
    return mocker.patch("tensorflow.Session")


@pytest.fixture()
def mocked_tf_placeholder(mocker):
    """Return mocked :func:`tensorflow.placeholder`."""
    return mocker.patch("tensorflow.placeholder")


@pytest.fixture()
def mocked_tf_config_proto(mocker):
    """Return mocked :func:`tensorflow.ConfigProto`."""
    return mocker.patch("tensorflow.ConfigProto")


@pytest.fixture()
def mocked_tf_reset_default_graph(mocker):
    """Return mocked :func:`tensorflow.reset_default_graph`."""
    return mocker.patch("tensorflow.reset_default_graph")


@pytest.fixture()
def mocked_tf_get_default_graph(mocker):
    """Return mocked :func:`tensorflow.get_default_graph`."""
    return mocker.patch("tensorflow.get_default_graph")


@pytest.fixture()
def mocked_core_create_session(mocker):
    """Return mocked :func:`stylish.core.create_session`."""
    return mocker.patch("stylish.core.create_session")


@pytest.fixture()
def mocked_filesystem_load_image(mocker):
    """Return mocked :func:`stylish.filesystem.load_image`."""
    return mocker.patch("stylish.filesystem.load_image")


@pytest.fixture()
def mocked_vgg_network(mocker):
    """Return mocked :func:`stylish.vgg.network`."""
    return mocker.patch("stylish.vgg.network")


def test_create_session(
    mocked_tf_session, mocked_tf_config_proto, mocked_tf_reset_default_graph,
):
    """Create a Tensorflow session."""
    mocked_config = mocked_tf_config_proto.return_value
    assert mocked_config.gpu_options.allow_growth is not True

    with stylish.core.create_session():
        mocked_tf_reset_default_graph.assert_called_once()
        mocked_tf_config_proto.assert_called_once_with(
            allow_soft_placement=True
        )
        assert mocked_config.gpu_options.allow_growth is True
        mocked_tf_session.assert_called_once_with(config=mocked_config)
        mocked_tf_session.return_value.close.assert_not_called()

    mocked_tf_session.return_value.close.assert_called_once()


@pytest.mark.parametrize("options, image_size", [
    ({}, None),
    ({"image_size": "__SIZE__"}, "__SIZE__"),
], ids=[
    "simple",
    "with-resize"
])
def test_extract_style_from_path(
    options, image_size, mocker, mocked_filesystem_load_image,
    mocked_vgg_network, mocked_core_create_session, mocked_tf_get_default_graph,
    mocked_tf_placeholder
):
    """Extract style feature mapping from image."""
    mocked_filesystem_load_image.return_value = np.ones((576, 720, 3))
    node = mocked_tf_placeholder.return_value - stylish.vgg.VGG19_MEAN
    mocked_graph = mocked_tf_get_default_graph.return_value
    mocked_graph.get_tensor_by_name.side_effect = ["TENSOR1", "TENSOR2"]
    mocked_session = (
         mocked_core_create_session.return_value.__enter__.return_value
    )
    mocked_session.run.side_effect = [
        np.ones((1, 576, 720, 64)),
        np.ones((1, 288, 360, 128))
    ]

    mapping = stylish.core.extract_style_from_path(
        "/path/to/style_image.jpg", "__VGG__",
        [("__LAYER_1__", 0.5), ("__LAYER_2__", 0.5)],
        **options
    )

    # Numpy arrays cannot be compared directly
    # https://github.com/pytest-dev/pytest/issues/5347
    assert sorted(mapping.keys()) == ["__LAYER_1__", "__LAYER_2__"]
    assert np.all(mapping["__LAYER_1__"] == np.full((64, 64), 0.0078125))
    assert np.all(mapping["__LAYER_2__"] == np.full((128, 128), 0.00390625))

    mocked_filesystem_load_image.assert_called_once_with(
        "/path/to/style_image.jpg", image_size=image_size
    )
    mocked_core_create_session.assert_called_once()

    mocked_tf_placeholder.assert_called_once_with(
        tf.float32, shape=(1, 576, 720, 3), name="input"
    )

    mocked_vgg_network.assert_called_once_with("__VGG__", node)
    mocked_tf_get_default_graph.assert_called_once()

    assert mocked_graph.get_tensor_by_name.call_count == 2
    mocked_graph.get_tensor_by_name.assert_any_call("vgg/__LAYER_1__:0")
    mocked_graph.get_tensor_by_name.assert_any_call("vgg/__LAYER_2__:0")

    assert mocked_session.run.call_count == 2
    mocked_session.run.assert_any_call("TENSOR1", feed_dict={node: mocker.ANY})
    mocked_session.run.assert_any_call("TENSOR2", feed_dict={node: mocker.ANY})

    # Numpy arrays cannot be compared directly
    # https://github.com/pytest-dev/pytest/issues/5347
    args = mocked_session.run.call_args_list
    assert np.all(args[0][1]["feed_dict"][node] == np.ones((1, 576, 720, 3)))
    assert np.all(args[1][1]["feed_dict"][node] == np.ones((1, 576, 720, 3)))
