# third-party imports
import pytest  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore

# module imports
from model import ConvolutionalBlock


@pytest.fixture(
    params=
    [
        (32, 64, 3, 1, False, None),
        (32, 64, 3, 1, True, 'prelu'),
        (32, 64, 5, 2, False, 'leakyrelu'),
        (32, 64, 5, 2, True, 'tanh'),
    ]
)
def conv_block_params(request):
    """Convolutional Block initialization params"""
    return request.param


def test_conv_block_output_shape(conv_block_params):
    in_channels, out_channels, kernel_size, stride, batch_norm, activation = conv_block_params
    conv_block = ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    batch_norm=batch_norm, activation=activation)
    dummy_input = tf.random.uniform((2, 128, 128, in_channels))
    out_channels = conv_block(dummy_input, training=False)
    assert out_channels.shape == (2, 128, 128, out_channels)
