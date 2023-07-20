# third-party imports
import pytest  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore
from PIL import Image  # type: ignore

# module imports
from transforms import ImageTransform


@pytest.fixture
def image_pil():
    """PIL Image"""
    return Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))


@pytest.fixture
def image_0_1():
    """Image in [0, 1]"""
    return tf.convert_to_tensor(np.random.random((32, 32, 3)), dtype=tf.float32)


@pytest.fixture
def image_minus1_1():
    return tf.convert_to_tensor(np.random.random((32, 32, 3)) * 2 - 1, dtype=tf.float32)


@pytest.fixture(name="config")
def transform_config():
    """Configuration parameters for the Image Transform instance"""
    config = {
        "split": "train",
        "crop_size": 16,
        "scaling_factor": 4,
        "lr_img_type": "pil",
        "hr_img_type": "pil",
    }
    return config


@pytest.fixture(name="transform")
def image_transform(config):
    """Image Transform"""
    image_tf = ImageTransform(**config)
    return image_tf


def test_image_transform_initialization(transform):
    """Tests the initialization of the Image Transform"""
    assert isinstance(transform, ImageTransform), "Object is not an ImageTransform instance"


@pytest.mark.parametrize(
    "source,target",
    [
        ('pil', '[0, 1]'), ('pil', '[-1, 1]'), ('pil', '[0, 255]'),
        ('[0, 1]', 'pil'), ('[0, 1]', '[-1, 1]'), ('[0, 1]', '[0, 255]'),
        ('[-1, 1]', 'pil'), ('[-1, 1]', '[0, 1]'), ('[-1, 1]', '[0, 255]')
    ]
)
def test_image_conversion(transform, image_pil, image_0_1, image_minus1_1, source, target):
    """Tests the conversion method."""

    img = None
    if source == "pil":
        img = image_pil
    elif source == "[0, 1]":
        img = image_0_1
    elif source == "[-1, 1]":
        img = image_minus1_1

    converted_img = transform.convert_image(img, source, target)

    if target == 'pil':
        assert isinstance(converted_img, Image.Image), f'Should be PIL Image, but got {type(converted_img)}'
    else:
        assert isinstance(converted_img, tf.Tensor), f'Should be Tensor, but got {type(converted_img)}'


def test_transform_call(transform, image_pil):
    """Tests the transform call."""

    lr_img, hr_img = transform(image_pil)

    assert hr_img.width == lr_img.width * transform.scaling_factor
    assert hr_img.height == lr_img.height * transform.scaling_factor

    if transform.lr_img_type == "pil":
        assert isinstance(lr_img, Image.Image)
    if transform.hr_img_type == "pil":
        assert isinstance(hr_img, Image.Image)



