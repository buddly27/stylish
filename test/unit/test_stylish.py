# :coding: utf-8

import os

import pytest

import stylish
import stylish.vgg


@pytest.fixture()
def mocked_vgg_extract_mapping(mocker):
    """Return mocked :func:`stylish.vgg.extract_mapping`."""
    return mocker.patch("stylish.vgg.extract_mapping")


@pytest.fixture()
def mocked_filesystem_create_log_path(mocker):
    """Return mocked :func:`stylish.filesystem.create_log_path`."""
    return mocker.patch("stylish.filesystem.create_log_path")


@pytest.fixture()
def mocked_filesystem_load_image(mocker):
    """Return mocked :func:`stylish.filesystem.load_image`."""
    return mocker.patch("stylish.filesystem.load_image")


@pytest.fixture()
def mocked_filesystem_fetch_images(mocker):
    """Return mocked :func:`stylish.filesystem.fetch_images`."""
    return mocker.patch("stylish.filesystem.fetch_images")


@pytest.fixture()
def mocked_filesystem_save_image(mocker):
    """Return mocked :func:`stylish.filesystem.save_image`."""
    return mocker.patch("stylish.filesystem.save_image")


@pytest.fixture()
def mocked_core_extract_style_from_path(mocker):
    """Return mocked :func:`stylish.core.extract_style_from_path`."""
    return mocker.patch("stylish.core.extract_style_from_path")


@pytest.fixture()
def mocked_core_optimize_image(mocker):
    """Return mocked :func:`stylish.core.optimize_image`."""
    return mocker.patch("stylish.core.optimize_image")


@pytest.fixture()
def mocked_core_optimize_model(mocker):
    """Return mocked :func:`stylish.core.optimize_model`."""
    return mocker.patch("stylish.core.optimize_model")


@pytest.fixture()
def mocked_core_infer_model(mocker):
    """Return mocked :func:`stylish.core.infer_model`."""
    return mocker.patch("stylish.core.infer_model")


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
            {"iterations": 999},
            999,
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
def test_transform_image(
    options, iterations, learning_rate, content_weight, style_weight, tv_weight,
    content_layer, style_layers, mocker, mocked_vgg_extract_mapping,
    mocked_filesystem_create_log_path, mocked_filesystem_load_image,
    mocked_filesystem_save_image, mocked_core_extract_style_from_path,
    mocked_core_optimize_image
):
    """Transfer style to another image."""
    mocked_image = mocker.Mock(shape="__SHAPE__")
    mocked_filesystem_load_image.return_value = mocked_image
    mocked_vgg_extract_mapping.return_value = "__VGG__"
    mocked_core_extract_style_from_path.return_value = "__STYLE__"
    mocked_filesystem_create_log_path.return_value = "__LOG__"
    mocked_core_optimize_image.return_value = "__OPTIMIZED_IMAGE__"

    image = stylish.transform_image(
        "/path/to/image.jpg",
        "/path/to/style_image.jpg",
        "/path/to/output_image",
        "/path/to/vgg_model.mat",
        **options
    )
    assert image == os.path.join("/path/to/output_image", "image.jpg")

    mocked_vgg_extract_mapping.assert_called_once_with("/path/to/vgg_model.mat")
    mocked_filesystem_load_image.assert_called_once_with("/path/to/image.jpg")
    mocked_filesystem_create_log_path.assert_called_once_with(
        "/path/to/style_image.jpg", relative_path="/path/to/output_image"
    )
    mocked_core_extract_style_from_path.assert_called_once_with(
        "/path/to/style_image.jpg", "__VGG__", style_layers,
        image_size="__SHAPE__"
    )

    mocked_core_optimize_image.assert_called_once_with(
        mocked_image, "__STYLE__", "__VGG__", "__LOG__",
        iterations,
        learning_rate=learning_rate,
        content_weight=content_weight,
        style_weight=style_weight,
        tv_weight=tv_weight,
        content_layer=content_layer,
        style_layer_names=[name for name, _ in style_layers]
    )

    mocked_filesystem_save_image.assert_called_once_with(
        "__OPTIMIZED_IMAGE__",
        os.path.join("/path/to/output_image", "image.jpg")
    )


def test_transform_image_with_log(
    mocker, mocked_vgg_extract_mapping, mocked_filesystem_create_log_path,
    mocked_filesystem_load_image, mocked_filesystem_save_image,
    mocked_core_extract_style_from_path, mocked_core_optimize_image
):
    """Transfer style to another image with specific log path."""
    mocked_image = mocker.Mock(shape="__SHAPE__")
    mocked_filesystem_load_image.return_value = mocked_image
    mocked_vgg_extract_mapping.return_value = "__VGG__"
    mocked_core_extract_style_from_path.return_value = "__STYLE__"
    mocked_core_optimize_image.return_value = "__OPTIMIZED_IMAGE__"

    image = stylish.transform_image(
        "/path/to/image.jpg",
        "/path/to/style_image.jpg",
        "/path/to/output_image",
        "/path/to/vgg_model.mat",
        log_path="__LOG__"
    )
    assert image == os.path.join("/path/to/output_image", "image.jpg")

    mocked_vgg_extract_mapping.assert_called_once_with("/path/to/vgg_model.mat")
    mocked_filesystem_load_image.assert_called_once_with("/path/to/image.jpg")
    mocked_filesystem_create_log_path.assert_not_called()
    mocked_core_extract_style_from_path.assert_called_once_with(
        "/path/to/style_image.jpg", "__VGG__", stylish.vgg.STYLE_LAYERS,
        image_size="__SHAPE__"
    )

    mocked_core_optimize_image.assert_called_once_with(
        mocked_image, "__STYLE__", "__VGG__", "__LOG__",
        stylish.core.ITERATIONS_NUMBER,
        learning_rate=stylish.core.LEARNING_RATE,
        content_weight=stylish.core.CONTENT_WEIGHT,
        style_weight=stylish.core.STYLE_WEIGHT,
        tv_weight=stylish.core.TV_WEIGHT,
        content_layer=stylish.vgg.CONTENT_LAYER,
        style_layer_names=[name for name, _ in stylish.vgg.STYLE_LAYERS]
    )

    mocked_filesystem_save_image.assert_called_once_with(
        "__OPTIMIZED_IMAGE__",
        os.path.join("/path/to/output_image", "image.jpg")
    )


@pytest.mark.parametrize(
    "options, learning_rate, batch_size, batch_shape, epoch_number, "
    "content_weight, style_weight, tv_weight, content_layer, style_layers, "
    "limit_training",
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
            None
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
            None
        ),
        (
            {"batch_size": 20},
            stylish.core.LEARNING_RATE,
            20,
            stylish.core.BATCH_SHAPE,
            stylish.core.EPOCHS_NUMBER,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS,
            None
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
            None
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
            None
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
            None
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
            None
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
            None
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
            None
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
            None
        ),
        (
            {"limit_training": 100},
            stylish.core.LEARNING_RATE,
            stylish.core.BATCH_SIZE,
            stylish.core.BATCH_SHAPE,
            stylish.core.EPOCHS_NUMBER,
            stylish.core.CONTENT_WEIGHT,
            stylish.core.STYLE_WEIGHT,
            stylish.core.TV_WEIGHT,
            stylish.vgg.CONTENT_LAYER,
            stylish.vgg.STYLE_LAYERS,
            100
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
        "with-limit-training",
    ]
)
def test_create_model(
    options, learning_rate, batch_size, batch_shape, epoch_number,
    content_weight, style_weight, tv_weight, content_layer, style_layers,
    limit_training, mocked_vgg_extract_mapping,
    mocked_filesystem_create_log_path, mocked_filesystem_fetch_images,
    mocked_core_extract_style_from_path, mocked_core_optimize_model
):
    """Train a style generator model."""
    mocked_filesystem_fetch_images.return_value = [
        "/path/to/image1", "/path/to/image2", "/path/to/image3"
    ]
    mocked_vgg_extract_mapping.return_value = "__VGG__"
    mocked_core_extract_style_from_path.return_value = "__STYLE__"
    mocked_filesystem_create_log_path.return_value = "__LOG__"

    stylish.create_model(
        "/path/to/training_data",
        "/path/to/style_image.jpg",
        "/path/to/output_image",
        "/path/to/vgg_model.mat",
        **options
    )

    mocked_vgg_extract_mapping.assert_called_once_with("/path/to/vgg_model.mat")
    mocked_filesystem_fetch_images.assert_called_once_with(
        "/path/to/training_data", limit=limit_training
    )
    mocked_filesystem_create_log_path.assert_called_once_with(
        "/path/to/style_image.jpg", relative_path="/path/to/output_image"
    )
    mocked_core_extract_style_from_path.assert_called_once_with(
        "/path/to/style_image.jpg", "__VGG__", style_layers
    )

    mocked_core_optimize_model.assert_called_once_with(
        ["/path/to/image1", "/path/to/image2", "/path/to/image3"],
        "__STYLE__", "__VGG__", "/path/to/output_image", "__LOG__",
        learning_rate=learning_rate,
        batch_size=batch_size,
        batch_shape=batch_shape,
        epoch_number=epoch_number,
        content_weight=content_weight,
        style_weight=style_weight,
        tv_weight=tv_weight,
        content_layer=content_layer,
        style_layer_names=[name for name, _ in style_layers]
    )


def test_create_model_with_log(
    mocked_vgg_extract_mapping, mocked_filesystem_create_log_path,
    mocked_filesystem_fetch_images, mocked_core_extract_style_from_path,
    mocked_core_optimize_model
):
    """Train a style generator model with specific log path."""
    mocked_filesystem_fetch_images.return_value = [
        "/path/to/image1", "/path/to/image2", "/path/to/image3"
    ]
    mocked_vgg_extract_mapping.return_value = "__VGG__"
    mocked_core_extract_style_from_path.return_value = "__STYLE__"

    stylish.create_model(
        "/path/to/training_data",
        "/path/to/style_image.jpg",
        "/path/to/output_image",
        "/path/to/vgg_model.mat",
        log_path="__LOG__"
    )

    mocked_vgg_extract_mapping.assert_called_once_with("/path/to/vgg_model.mat")
    mocked_filesystem_fetch_images.assert_called_once_with(
        "/path/to/training_data", limit=None
    )
    mocked_filesystem_create_log_path.assert_not_called()
    mocked_core_extract_style_from_path.assert_called_once_with(
        "/path/to/style_image.jpg", "__VGG__", stylish.vgg.STYLE_LAYERS
    )

    mocked_core_optimize_model.assert_called_once_with(
        ["/path/to/image1", "/path/to/image2", "/path/to/image3"],
        "__STYLE__", "__VGG__", "/path/to/output_image", "__LOG__",
        learning_rate=stylish.core.LEARNING_RATE,
        batch_size=stylish.core.BATCH_SIZE,
        batch_shape=stylish.core.BATCH_SHAPE,
        epoch_number=stylish.core.EPOCHS_NUMBER,
        content_weight=stylish.core.CONTENT_WEIGHT,
        style_weight=stylish.core.STYLE_WEIGHT,
        tv_weight=stylish.core.TV_WEIGHT,
        content_layer=stylish.vgg.CONTENT_LAYER,
        style_layer_names=[name for name, _ in stylish.vgg.STYLE_LAYERS]
    )


def test_apply_model(mocked_filesystem_save_image, mocked_core_infer_model):
    """Apply style generator model to a new image."""
    mocked_core_infer_model.return_value = "__GENERATED_IMAGE__"

    image = stylish.apply_model(
        "/path/to/saved_model",
        "/path/to/image.jpg",
        "/path/to/output_image"
    )
    assert image == os.path.join("/path/to/output_image", "image.jpg")

    mocked_filesystem_save_image.assert_called_once_with(
        "__GENERATED_IMAGE__",
        os.path.join("/path/to/output_image", "image.jpg")
    )

    mocked_core_infer_model.assert_called_once_with(
        "/path/to/saved_model", "/path/to/image.jpg"
    )


@pytest.mark.parametrize(
    "options, style_layers",
    [
        (
            {},
            stylish.vgg.STYLE_LAYERS
        ),
        (
            {"style_layers": [("__LAYER_1__", 1), ("__LAYER_2__", 2)]},
            [("__LAYER_1__", 1), ("__LAYER_2__", 2)]
        )
    ],
    ids=[
        "simple",
        "with-style-layers",
    ]
)
def test_extract_style_pattern(
    options, style_layers, mocker, mocked_vgg_extract_mapping,
    mocked_filesystem_load_image, mocked_core_extract_style_from_path,
    mocked_filesystem_save_image
):
    """Generate style pattern images from image."""
    mocked_image = mocker.Mock(shape="__SHAPE__")
    mocked_filesystem_load_image.return_value = mocked_image
    mocked_vgg_extract_mapping.return_value = "__VGG__"
    mocked_core_extract_style_from_path.return_value = {
        "LAYER1": "__IMAGE1__",
        "LAYER2": "__IMAGE2__",
        "LAYER3": "__IMAGE3__",
    }

    images = stylish.extract_style_pattern(
        "/path/to/style_image.jpg",
        "/path/to/output",
        "/path/to/vgg_model.mat",
        **options
    )
    assert images == [
        "/path/to/output/style_image_LAYER1.jpg",
        "/path/to/output/style_image_LAYER2.jpg",
        "/path/to/output/style_image_LAYER3.jpg",
    ]

    mocked_vgg_extract_mapping.assert_called_once_with("/path/to/vgg_model.mat")
    mocked_filesystem_load_image.assert_called_once_with(
        "/path/to/style_image.jpg"
    )
    mocked_core_extract_style_from_path.assert_called_once_with(
        "/path/to/style_image.jpg", "__VGG__", style_layers,
        image_size="__SHAPE__"
    )

    assert mocked_filesystem_save_image.call_count == 3
    mocked_filesystem_save_image.assert_any_call(
        "__IMAGE1__", "/path/to/output/style_image_LAYER1.jpg"
    )
    mocked_filesystem_save_image.assert_any_call(
        "__IMAGE2__", "/path/to/output/style_image_LAYER2.jpg"
    )
    mocked_filesystem_save_image.assert_any_call(
        "__IMAGE3__", "/path/to/output/style_image_LAYER3.jpg"
    )
