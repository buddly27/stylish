# :coding: utf-8

import pytest
import tensorflow as tf
import numpy as np

import stylish.core


@pytest.fixture()
def mocked_tf_session(mocker):
    """Return mocked :class:`tensorflow.Session`."""
    return mocker.patch("tensorflow.Session")


@pytest.fixture()
def mocked_tf_placeholder(mocker):
    """Return mocked :func:`tensorflow.placeholder`."""
    return mocker.patch("tensorflow.placeholder")


@pytest.fixture()
def mocked_tf_identity(mocker):
    """Return mocked :func:`tensorflow.identity`."""
    return mocker.patch("tensorflow.identity")


@pytest.fixture()
def mocked_tf_config_proto(mocker):
    """Return mocked :class:`tensorflow.ConfigProto`."""
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
def mocked_tf_global_variables_initializer(mocker):
    """Return mocked :func:`tensorflow.global_variables_initializer`."""
    return mocker.patch("tensorflow.global_variables_initializer")


@pytest.fixture()
def mocked_tf_train(mocker):
    """Return mocked :mod:`tensorflow.train`."""
    return mocker.patch("tensorflow.train")


@pytest.fixture()
def mocked_tf_summary(mocker):
    """Return mocked :mod:`tensorflow.summary`."""
    return mocker.patch("tensorflow.summary")


@pytest.fixture()
def mocked_core_create_session(mocker):
    """Return mocked :func:`stylish.core.create_session`."""
    return mocker.patch("stylish.core.create_session")


@pytest.fixture()
def mocked_core_compute_cost(mocker):
    """Return mocked :func:`stylish.core.compute_cost`."""
    return mocker.patch("stylish.core.compute_cost")


@pytest.fixture()
def mocked_core_get_next_batch(mocker):
    """Return mocked :func:`stylish.core.get_next_batch`."""
    return mocker.patch("stylish.core.get_next_batch")


@pytest.fixture()
def mocked_core_save_model(mocker):
    """Return mocked :func:`stylish.core.save_model`."""
    return mocker.patch("stylish.core.save_model")


@pytest.fixture()
def mocked_filesystem_load_image(mocker):
    """Return mocked :func:`stylish.filesystem.load_image`."""
    return mocker.patch("stylish.filesystem.load_image")


@pytest.fixture()
def mocked_vgg_network(mocker):
    """Return mocked :func:`stylish.vgg.network`."""
    return mocker.patch("stylish.vgg.network")


@pytest.fixture()
def mocked_transform_network(mocker):
    """Return mocked :func:`stylish.transform.network`."""
    return mocker.patch("stylish.transform.network")


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

    # Run command to test
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


@pytest.mark.parametrize(
    "options, iterations, learning_rate, content_weight, style_weight, "
    "tv_weight, content_layer, style_layers",
    [
        (
            {},
            stylish.core.ITERATIONS_NUMBER,
            stylish.core.LEARNING_RATE,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS
        ),
        (
            {"iterations": 1},
            1,
            stylish.core.LEARNING_RATE,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS
        ),
        (
            {"learning_rate": 0.99},
            stylish.core.ITERATIONS_NUMBER,
            0.99,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS
        ),
        (
            {"content_weight": 0.99},
            stylish.core.ITERATIONS_NUMBER,
            stylish.core.LEARNING_RATE,
            0.99,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS
        ),
        (
            {"style_weight": 0.99},
            stylish.core.ITERATIONS_NUMBER,
            stylish.core.LEARNING_RATE,
            stylish.core.CONTENT_WEIGHT,
            0.99,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS
        ),
        (
            {"tv_weight": 0.99},
            stylish.core.ITERATIONS_NUMBER,
            stylish.core.LEARNING_RATE,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            0.99,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS
        ),
        (
            {"content_layer": "__CONTENT_LAYER__"},
            stylish.core.ITERATIONS_NUMBER,
            stylish.core.LEARNING_RATE,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            "__CONTENT_LAYER__",
            stylish.vgg.STYLE_LAYERS
        ),
        (
            {"style_layers": [("__LAYER_1__", 1), ("__LAYER_2__", 2)]},
            stylish.core.ITERATIONS_NUMBER,
            stylish.core.LEARNING_RATE,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            [("__LAYER_1__", 1), ("__LAYER_2__", 2)]
        )
    ],
    ids=[
        "simple",
        "with-iterations",
        "with-learning-rate",
        "with-content-weight",
        "with-style-weight",
        "with-tv-weight",
        "with-content-layer",
        "with-style-layers",
    ]
)
def test_optimize_image(
    options, iterations, learning_rate, content_weight, style_weight,
    tv_weight, content_layer, style_layers, mocker, mocked_core_create_session,
    mocked_transform_network, mocked_vgg_network, mocked_core_compute_cost,
    mocked_tf_summary, mocked_tf_train, mocked_tf_placeholder,
    mocked_tf_global_variables_initializer
):
    """Transfer style mapping features to image."""
    # Update default iterations to 10 to improve speed
    mocker.patch("stylish.core.ITERATIONS_NUMBER", 4)

    input_node = mocked_tf_placeholder.return_value
    output_node = mocked_transform_network.return_value

    mocked_core_compute_cost.return_value = "__COST__"
    mocked_session = (
         mocked_core_create_session.return_value.__enter__.return_value
    )

    # Returned value for global variables initializer
    session_results = [None]

    # Returned values for each iteration
    session_results += [(None, "__SUMMARY__")] * iterations

    # Returned value for final image
    session_results += [["__GENERATED_IMAGE__"]]

    mocked_session.run.side_effect = session_results

    # Run command to test
    image = stylish.core.optimize_image(
        np.ones((576, 720, 3)), "__STYLE__", "__VGG__", "/path/to/log",
        **options
    )
    assert image == "__GENERATED_IMAGE__"

    mocked_core_create_session.assert_called_once()
    mocked_tf_placeholder.assert_called_once_with(
        tf.float32, shape=(1, 576, 720, 3), name="input"
    )

    mocked_transform_network.assert_called_once_with(
        (input_node - stylish.vgg.VGG19_MEAN) / 255.0
    )

    assert mocked_vgg_network.call_count == 2
    mocked_vgg_network.assert_any_call(
        "__VGG__", input_node - stylish.vgg.VGG19_MEAN
    )
    mocked_vgg_network.assert_any_call(
        "__VGG__", output_node - stylish.vgg.VGG19_MEAN
    )

    mocked_core_compute_cost.assert_called_once_with(
        mocked_session, "__STYLE__", output_node,
        batch_size=1,
        content_weight=content_weight,
        style_weight=style_weight,
        tv_weight=tv_weight,
        content_layer=content_layer,
        style_layers=[name for name, _ in style_layers],
        input_namespace="vgg1",
        output_namespace="vgg2"
    )

    mocked_tf_global_variables_initializer.assert_called_once()
    mocked_tf_train.AdamOptimizer.assert_called_once_with(learning_rate)
    mocked_tf_train.AdamOptimizer.return_value.minimize("__COST__")

    assert mocked_session.run.call_count == iterations + 2
    mocked_session.run.assert_any_call(
        mocked_tf_global_variables_initializer.return_value
    )
    mocked_session.run.assert_any_call(
        [
            mocked_tf_train.AdamOptimizer.return_value.minimize.return_value,
            mocked_tf_summary.merge_all.return_value
        ],
        feed_dict={input_node: mocker.ANY}
    )
    mocked_session.run.assert_any_call(
        output_node, feed_dict={input_node: mocker.ANY}
    )

    # Numpy arrays cannot be compared directly
    # https://github.com/pytest-dev/pytest/issues/5347
    args = mocked_session.run.call_args_list
    for index in range(iterations + 1):
        input_image = args[index + 1][1]["feed_dict"][input_node]
        assert np.all(input_image == np.ones((1, 576, 720, 3)))


@pytest.mark.parametrize(
    "options, learning_rate, batch_size, batch_shape, epoch_number, "
    "content_weight, style_weight, tv_weight, content_layer, style_layers",
    [
        (
            {},
            stylish.core.LEARNING_RATE,
            stylish.core.BATCH_SIZE,
            stylish.core.BATCH_SHAPE,
            stylish.core.EPOCHS_NUMBER,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS,
        ),
        (
            {"learning_rate": 0.99},
            0.99,
            stylish.core.BATCH_SIZE,
            stylish.core.BATCH_SHAPE,
            stylish.core.EPOCHS_NUMBER,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS,
        ),
        (
            {"batch_size": 1},
            stylish.core.LEARNING_RATE,
            1,
            stylish.core.BATCH_SHAPE,
            stylish.core.EPOCHS_NUMBER,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS,
        ),
        (
            {"batch_shape": (64, 64, 3)},
            stylish.core.LEARNING_RATE,
            stylish.core.BATCH_SIZE,
            (64, 64, 3),
            stylish.core.EPOCHS_NUMBER,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS,
        ),
        (
            {"epoch_number": 99},
            stylish.core.LEARNING_RATE,
            stylish.core.BATCH_SIZE,
            stylish.core.BATCH_SHAPE,
            99,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS,
        ),
        (
            {"content_weight": 0.99},
            stylish.core.LEARNING_RATE,
            stylish.core.BATCH_SIZE,
            stylish.core.BATCH_SHAPE,
            stylish.core.EPOCHS_NUMBER,
            0.99,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS,
        ),
        (
            {"style_weight": 0.99},
            stylish.core.LEARNING_RATE,
            stylish.core.BATCH_SIZE,
            stylish.core.BATCH_SHAPE,
            stylish.core.EPOCHS_NUMBER,
            stylish.core.CONTENT_WEIGHT,
            0.99,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS,
        ),
        (
            {"tv_weight": 0.99},
            stylish.core.LEARNING_RATE,
            stylish.core.BATCH_SIZE,
            stylish.core.BATCH_SHAPE,
            stylish.core.EPOCHS_NUMBER,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            0.99,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS,
        ),
        (
            {"content_layer": "__CONTENT_LAYER__"},
            stylish.core.LEARNING_RATE,
            stylish.core.BATCH_SIZE,
            stylish.core.BATCH_SHAPE,
            stylish.core.EPOCHS_NUMBER,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            "__CONTENT_LAYER__",
            stylish.vgg.STYLE_LAYERS,
        ),
        (
            {"style_layers": [("__LAYER_1__", 1), ("__LAYER_2__", 2)]},
            stylish.core.LEARNING_RATE,
            stylish.core.BATCH_SIZE,
            stylish.core.BATCH_SHAPE,
            stylish.core.EPOCHS_NUMBER,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            [("__LAYER_1__", 1), ("__LAYER_2__", 2)],
        )
    ],
    ids=[
        "simple",
        "with-learning-rate",
        "with-batch-size",
        "with-batch-shape",
        "with-epoch-number",
        "with-content-weight",
        "with-style-weight",
        "with-tv-weight",
        "with-content-layer",
        "with-style-layers",
    ]
)
def test_optimize_model(
    options, learning_rate, batch_size, batch_shape, epoch_number,
    content_weight, style_weight, tv_weight, content_layer, style_layers,
    mocker, mocked_core_create_session, mocked_transform_network,
    mocked_vgg_network, mocked_core_compute_cost, mocked_tf_summary,
    mocked_tf_train, mocked_tf_placeholder, mocked_tf_identity,
    mocked_tf_global_variables_initializer, mocked_core_get_next_batch,
    mocked_core_save_model

):
    """Create style generator model from a style mapping and a training dataset.
    """
    training_images = [
        "/path/to/image1",
        "/path/to/image2",
        "/path/to/image3",
        "/path/to/image4",
        "/path/to/image5",
    ]

    input_node = mocked_tf_placeholder.return_value
    output_node = mocked_tf_identity.return_value

    mocked_core_compute_cost.return_value = "__COST__"
    mocked_core_get_next_batch.return_value = "__IMAGES_BATCH__"
    mocked_session = (
         mocked_core_create_session.return_value.__enter__.return_value
    )

    # Returned value for global variables initializer
    session_results = [None]

    # Returned values for each iteration
    session_results += (
        [(None, "__SUMMARY__")]
        * (len(training_images) // batch_size)
        * epoch_number
    )

    mocked_session.run.side_effect = session_results

    # Run command to test
    stylish.core.optimize_model(
        training_images,
        "__STYLE__", "__VGG__",
        "/path/to/output_model",
        "/path/to/log",
        **options
    )

    mocked_core_create_session.assert_called_once()
    mocked_tf_placeholder.assert_called_once_with(
        tf.float32, shape=(None, None, None, None), name="input"
    )
    mocked_transform_network.assert_called_once_with(
        (input_node - stylish.vgg.VGG19_MEAN) / 255.0
    )
    mocked_tf_identity.assert_called_once_with(
        mocked_transform_network.return_value, name="output"
    )

    assert mocked_vgg_network.call_count == 2
    mocked_vgg_network.assert_any_call(
        "__VGG__", input_node - stylish.vgg.VGG19_MEAN
    )
    mocked_vgg_network.assert_any_call(
        "__VGG__", output_node - stylish.vgg.VGG19_MEAN
    )

    mocked_core_compute_cost.assert_called_once_with(
        mocked_session, "__STYLE__", output_node,
        batch_size=batch_size,
        content_weight=content_weight,
        style_weight=style_weight,
        tv_weight=tv_weight,
        content_layer=content_layer,
        style_layers=[name for name, _ in style_layers],
        input_namespace="vgg1",
        output_namespace="vgg2"
    )

    mocked_tf_global_variables_initializer.assert_called_once()
    mocked_tf_train.AdamOptimizer.assert_called_once_with(learning_rate)
    mocked_tf_train.AdamOptimizer.return_value.minimize("__COST__")

    assert mocked_session.run.call_count == (
        (len(training_images) // batch_size) * epoch_number + 1
    )
    mocked_session.run.assert_any_call(
        mocked_tf_global_variables_initializer.return_value
    )
    mocked_session.run.assert_any_call(
        [
            mocked_tf_train.AdamOptimizer.return_value.minimize.return_value,
            mocked_tf_summary.merge_all.return_value
        ],
        feed_dict={input_node: "__IMAGES_BATCH__"}
    )

    assert mocked_core_get_next_batch.call_count == (
        (len(training_images) // batch_size)
        * epoch_number
    )
    mocked_core_get_next_batch.assert_any_call(
        mocker.ANY, training_images,
        batch_size=batch_size,
        batch_shape=batch_shape
    )

    mocked_core_save_model.assert_called_once_with(
        mocked_session, input_node, output_node, "/path/to/output_model"
    )
