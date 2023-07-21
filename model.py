# standard imports
import math
from typing import List, Optional

# third-party imports
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, Model  # type: ignore


class ConvolutionalBlock(layers.Layer):
    """Convolutional Block with Batch Norm and customizable activation layers"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        batch_norm: bool = False,
        activation: Optional[str] = None,
        **kwargs,
    ):
        """
        Convolutional Block initialization.

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: conv filter size
        :param stride: stride step in the conv filter
        :param batch_norm: whether to include batch normalization or not
        :param activation: type of activation, optional, none by default
        """

        super().__init__(**kwargs)

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}, f"{activation} not implemented"

        blocks: List[layers.Layer] = []

        # PReLU and LeakyReLU have configurable parameters, so we can't just pass the strings to Keras
        if activation == 'prelu':
            self.activation_layer = layers.PReLU()
        elif activation == 'leakyrelu':
            self.activation_layer = layers.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation_layer = layers.Activation(tf.keras.activations.tanh)

        # O = (W - K + 2P) / S + 1, so padding='same' is the same as padding = kernel_size // 2 and stride = 1
        blocks.append(layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride,
                                    padding='same', input_shape=(None, None, in_channels)))

        if batch_norm:
            blocks.append(layers.BatchNormalization())

        self.conv_block = tf.keras.Sequential(blocks)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass of the Convolutional Block.

        :param inputs: input images, a Tensor of shape (N, W, H, in_channels)
        :param training: whether the layer is in training mode or not
        :return: output images, a Tensor of shape (N, W, H, out_channels)
        """

        output = self.conv_block(inputs, training=training)  # (N, W, H, out_channels)
        if hasattr(self, 'activation_layer'):
            output = self.activation_layer(output)
        return output


class SubPixelConvolutionalBlock(layers.Layer):
    """Subpixel Conv Block mapping depth to space (pixel shuffling) with convolutional layers."""

    def __init__(self, kernel_size: int = 3, n_channels:int = 64, scaling_factor: int = 2, **kwargs):
        """
        Initializes the Sub Pixel Conv Block.

        :param kernel_size: conv filter size
        :param n_channels: number of both input and output channels
        :param scaling_factor: factor to scale the input images by (along both dimensions)
        """
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters=n_channels * (scaling_factor ** 2),
                                  kernel_size=kernel_size, padding='same')
        self.scaling_factor = scaling_factor
        self.prelu = layers.PReLU()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass

        :param inputs: input images, a Tensor of shape (N, w, h, n_channels)
        :return: scaled output images, a Tensor of shape (N, w * scaling_factor, h * scaling_factor, n_channels)
        """
        output = self.conv(inputs)  # (N, w, h, n_channels * scaling_factor)
        output = tf.nn.depth_to_space(output, self.scaling_factor, data_format='NHWC')
        output = self.prelu(output)
        return output


class ResidualBlock(layers.Layer):
    """Two conv blocks stacked together with a residual connection across them."""

    def __init__(self, kernel_size: int = 3, n_channels: int = 64, **kwargs):
        """
        Initializes the Residual Block.

        :param kernel_size: conv filter size
        :param n_channels: number of both input and output channels
        """
        super().__init__(**kwargs)
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=kernel_size, batch_norm=True,
                                              activation='prelu')
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=kernel_size, batch_norm=True,
                                              activation=None)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass.

        :param inputs: input images, a Tensor of shape (N, w, h , n_channels)
        :param training: whether the layer is in training mode or not
        :return: output images, a Tensor of shape (N, w, h, n_channels)
        """

        output = self.conv_block1(inputs, training=training)
        output = self.conv_block2(output, training=training)
        output += inputs

        return output


class SuperResolutionResNet(Model):
    """Super Resolution ResNet"""

    def __init__(
        self,
        large_kernel_size: int = 9,
        small_kernel_size: int = 3,
        n_channels: int = 64,
        n_blocks: int = 16,
        scaling_factor: int = 4,
        **kwargs
    ):
        """
        Initializes the Super Resolution Resnet
        :param large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
        :param small_kernel_size: kernel size of all convolutions in-between (residual & subpixel convolutional blocks)
        :param n_channels: number of channels in-between, input and output channels for residual & subpixel conv blocks
        :param n_blocks: number of residual blocks
        :param scaling_factor: factor to scale input images by (along both dimensions) in the subpixel conv block
        """
        super().__init__(**kwargs)
        assert scaling_factor in {2, 4, 8}, "The scaling factor must be 2, 4, or 8!"

        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels,
                                              kernel_size=large_kernel_size, batch_norm=False,
                                              activation='prelu')
        self.residual_blocks = tf.keras.Sequential(
            [ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for _ in range(n_blocks)]
        )
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size, batch_norm=False,
                                              activation=None)

        n_subpixel_conv_blocks = int(math.log2(scaling_factor))
        # up-scaling by a factor of 2 (so squaring)
        self.subpixel_conv_blocks = tf.keras.Sequential(
            [SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for _
            in range(n_subpixel_conv_blocks)]
        )
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3,
                                              kernel_size=large_kernel_size, batch_norm=False,
                                              activation='tanh')

    def call(self, low_res_images: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass.

        :param low_res_images: low resolution input images, a Tensor of shape (N, w, h, 3)
        :param training: whether the layer is in training mode or not
        :return:
        """
        output = self.conv_block1(low_res_images, training=training)
        residual = output
        output = self.residual_blocks(output, training=training)
        output = self.conv_block2(output, training=training)
        output = output + residual
        output = self.subpixel_conv_blocks(output)
        super_res_images = self.conv_block3(output, training=training)

        return super_res_images
