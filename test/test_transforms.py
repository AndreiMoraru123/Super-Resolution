# third-party imports
import pytest  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore
from PIL import Image  # type: ignore

# module imports
from transforms import ImageTransform


@pytest.fixture
def image_pil():
    """PIL Image."""
    return Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))


@pytest.fixture
def image_0_1():
    """Image in [0, 1]."""
    return tf.convert_to_tensor(np.random.random((32, 32, 3)), dtype=tf.float32)


@pytest.fixture
def image_minus1_1():
    """Image in [1, 1]."""
    return tf.convert_to_tensor(np.random.random((32, 32, 3)) * 2 - 1, dtype=tf.float32)


@pytest.fixture(name="pil_config")
def transform_config_pil():
    """Configuration parameters for the Image Transform instance (PIL)."""
    config = {
        "split": "train",
        "crop_size": 16,
        "scaling_factor": 4,
        "lr_img_type": "pil",
        "hr_img_type": "pil",
    }
    return config


@pytest.fixture(name="tensor_config")
def transform_config_tensor():
    """Configuration parameters for the Image Transform instance (Tensor)."""
    config = {
        "split": "train",
        "crop_size": 16,
        "scaling_factor": 4,
        "lr_img_type": "imagenet-norm",
        "hr_img_type": "[-1, 1]",
    }
    return config


@pytest.fixture(name="pil_transform")
def image_transform_pil(pil_config):
    """Image Transform in pil config."""
    image_tf = ImageTransform(**pil_config)
    return image_tf


@pytest.fixture(name="tensor_transform")
def image_transform_tensor(tensor_config):
    """Image Transform in tensor config."""
    image_tf = ImageTransform(**tensor_config)
    return image_tf


def test_image_transform_initialization_pil(pil_transform):
    """Tests the initialization of the Image Transform."""
    assert isinstance(
        pil_transform, ImageTransform
    ), "Object is not an ImageTransform instance"


def test_image_transform_initialization_tensor(tensor_transform):
    """Tests the initialization of the Image Transform."""
    assert isinstance(
        tensor_transform, ImageTransform
    ), "Object is not an ImageTransform instance"


@pytest.mark.parametrize(
    "source,target",
    [
        ("pil", "[0, 1]"),
        ("pil", "[-1, 1]"),
        ("pil", "[0, 255]"),
        ("[0, 1]", "pil"),
        ("[0, 1]", "[-1, 1]"),
        ("[0, 1]", "[0, 255]"),
        ("[-1, 1]", "pil"),
        ("[-1, 1]", "[0, 1]"),
        ("[-1, 1]", "[0, 255]"),
    ],
)
def test_image_conversion(
    pil_transform, image_pil, image_0_1, image_minus1_1, source, target
):
    """Tests the conversion method."""

    img = None
    if source == "pil":
        img = image_pil
    elif source == "[0, 1]":
        img = image_0_1
    elif source == "[-1, 1]":
        img = image_minus1_1

    converted_img = pil_transform.convert_image(img=img, source=source, target=target)

    if target == "pil":
        assert isinstance(
            converted_img, Image.Image
        ), f"Should be PIL Image, but got {type(converted_img)}"
    else:
        assert isinstance(
            converted_img, tf.Tensor
        ), f"Should be Tensor, but got {type(converted_img)}"


def test_transform_call(pil_transform, tensor_transform, image_pil):
    """Tests the transform call."""

    lr_img, hr_img = pil_transform(image_pil)

    assert isinstance(lr_img, Image.Image)
    assert isinstance(hr_img, Image.Image)

    lr_img, hr_img = tensor_transform(image_pil)

    assert isinstance(lr_img, tf.Tensor)
    assert isinstance(hr_img, tf.Tensor)
