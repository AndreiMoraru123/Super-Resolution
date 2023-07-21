# standard imports
from typing import List, Optional

# third-party imports
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras import Model  # type: ignore


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

        super(ConvolutionalBlock, self).__init__(**kwargs)

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

