# third-party imports
import pytest  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore

# module imports
from model import ConvolutionalBlock, SubPixelConvolutionalBlock, ResidualBlock


@pytest.fixture(
    params=
    [
        (32, 64, 3, 1, False, None),
        (32, 64, 3, 1, True, 'prelu'),
        (32, 64, 5, 1, False, 'leakyrelu'),
        (32, 64, 5, 1, True, 'tanh'),
    ]
)
def conv_block_params(request):
    """Convolutional Block initialization params"""
    return request.param


@pytest.fixture(
    params=
    [
        (3, 64, 2),
        (5, 32, 3),
        (3, 128, 4),
        (5, 256, 2),
    ]
)
def subpixel_conv_block_params(request):
    """Sub Pixel Convolutional Block initialization params"""
    return request.param


@pytest.fixture(
    params=
    [
        (3, 64),
        (5, 32),
        (3, 128),
        (5, 256),
    ]
)
def residual_block_params(request):
    """Residual Block initialization params"""
    return request.param


def test_conv_block_output_shape(conv_block_params):
    in_channels, out_channels, kernel_size, stride, batch_norm, activation = conv_block_params
    conv_block = ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    batch_norm=batch_norm, activation=activation)
    dummy_input = tf.random.uniform((2, 128, 128, in_channels))
    output = conv_block(dummy_input, training=False)
    assert output.shape == (2, 128, 128, out_channels)


def test_subpixel_conv_block_output_shape(subpixel_conv_block_params):
    kernel_size, n_channels, scaling_factor = subpixel_conv_block_params
    subpixel_conv_block = SubPixelConvolutionalBlock(kernel_size=kernel_size,
                                                     n_channels=n_channels,
                                                     scaling_factor=scaling_factor)
    dummy_input = tf.random.uniform((2, 128, 128, n_channels))
    output = subpixel_conv_block(dummy_input)
    assert output.shape == (2, 128 * scaling_factor, 128 * scaling_factor, n_channels)


def test_residual_block_output_shape(residual_block_params):
    kernel_size, n_channels = residual_block_params
    residual_block = ResidualBlock(kernel_size=kernel_size, n_channels=n_channels)
    dummy_input = tf.random.uniform((2, 128, 128, n_channels))
    output = residual_block(dummy_input)
    assert output.shape == dummy_input.shape
