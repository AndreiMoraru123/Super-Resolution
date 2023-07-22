# standard imports
import unittest.mock as mock

# third-party imports
import pytest  # type: ignore
import tensorflow as tf  # type: ignore

# module imports
from model import (ConvolutionalBlock, SubPixelConvolutionalBlock,
                   ResidualBlock, SuperResolutionResNet,
                   Generator, Discriminator,
                   TruncatedVGG19)


class MockTensor:
    """Mocking a Tensor to get access to its dimensions"""
    def __init__(self, shape):
        self.shape = shape


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


@pytest.fixture(
    params=
    [
        (9, 3, 64, 16, 2),
        (9, 3, 64, 16, 4),
        (9, 3, 64, 16, 8),
        (8, 5, 64, 16, 4),
    ]
)
def sr_resnet_params(request):
    """Residual Block initialization params"""
    return request.param


@pytest.fixture(
    params=
    [
        (3, 64, 8, 1024),
        (3, 32, 6, 512),
        (3, 16, 4, 2048),
        (3, 8, 20, 256),
    ]
)
def discriminator_params(request):
    """Residual Block initialization params"""
    return request.param


@pytest.fixture(
    params=
    [
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 4)
    ]
)
def vgg_indices(request):
    """Parameters for the VGG indices i & j."""
    return request.param


def test_conv_block_output_shape(conv_block_params):
    """Conv Block forward pass to maintain the shape and change the number of channels."""
    in_channels, out_channels, kernel_size, stride, batch_norm, activation = conv_block_params
    conv_block = ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    batch_norm=batch_norm, activation=activation)
    dummy_input = tf.random.uniform((2, 128, 128, in_channels))
    output = conv_block(dummy_input, training=False)
    assert output.shape == (2, 128, 128, out_channels)


def test_subpixel_conv_block_output_shape(subpixel_conv_block_params):
    """Sub Pixel Conv Block forward pass to scale the image by the given factor."""
    kernel_size, n_channels, scaling_factor = subpixel_conv_block_params
    subpixel_conv_block = SubPixelConvolutionalBlock(kernel_size=kernel_size,
                                                     n_channels=n_channels,
                                                     scaling_factor=scaling_factor)
    dummy_input = tf.random.uniform((2, 128, 128, n_channels))
    output = subpixel_conv_block(dummy_input)
    assert output.shape == (2, 128 * scaling_factor, 128 * scaling_factor, n_channels)


def test_residual_block_output_shape(residual_block_params):
    """Residual Block forward pass to keep the same shape till the end."""
    kernel_size, n_channels = residual_block_params
    residual_block = ResidualBlock(kernel_size=kernel_size, n_channels=n_channels)
    dummy_input = tf.random.uniform((2, 128, 128, n_channels))
    output = residual_block(dummy_input)
    assert output.shape == dummy_input.shape


def test_sr_resnet_output_shape(sr_resnet_params):
    """Super Resolution ResNet forward pass to output an up-scaled image."""
    large_kernel_size, small_kernel_size, n_channels, n_blocks, scaling_factor = sr_resnet_params
    sr_resnet = SuperResolutionResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                                      n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)
    dummy_input = tf.random.uniform((2, 16, 16, 3))
    output = sr_resnet(dummy_input)
    assert output.shape == (2, 16 * scaling_factor, 16 * scaling_factor, 3)


def test_generator_output_shape(sr_resnet_params):
    """Super Resolution GAN Generator forward pass to output an up-scaled image."""
    large_kernel_size, small_kernel_size, n_channels, n_blocks, scaling_factor = sr_resnet_params
    generator = Generator(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                          n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)

    dummy_input = tf.random.uniform((2, 16, 16, 3))
    mock_output_tensor = MockTensor(shape=(dummy_input.shape[0], dummy_input.shape[1] * scaling_factor,
                                           dummy_input.shape[2] * scaling_factor, dummy_input.shape[3]))

    mock_sr_resnet_model = mock.Mock()
    mock_sr_resnet_model.call.return_value = mock_output_tensor

    # We know the generator is behaving like ResNet, so we mock it just for fun
    with mock.patch('tensorflow.keras.models.load_model', return_value=mock_sr_resnet_model):
        generator.initialize_with_srresnet('mock_srresnet_checkpoint')

    # I don't actually write code like this in real life
    generator.call = mock.Mock(return_value=mock_output_tensor)

    output = generator(dummy_input)
    assert output.shape == (2, 16 * scaling_factor, 16 * scaling_factor, 3)


def test_discriminator_output_shape(discriminator_params):
    """Discriminator forward pass to output a logit."""
    discriminator = Discriminator(*discriminator_params)
    dummy_input = tf.random.uniform((2, 16 * 4, 16 * 4, 3))
    output = discriminator(dummy_input)
    assert tf.rank(output) == 1


def test_truncated_vgg19_output(vgg_indices):
    """Testing the Truncated VGG for feature extraction."""
    i, j = vgg_indices
    truncated_vgg19 = TruncatedVGG19(i, j)

    dummy_input = tf.random.uniform((2, 224, 224, 3))
    dummy_input = tf.keras.applications.vgg19.preprocess_input(dummy_input)

    output = truncated_vgg19(dummy_input)
    assert tf.rank(output) == tf.rank(dummy_input)
    assert output.shape[0] == dummy_input.shape[0]
