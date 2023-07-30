# third-party imports
import tensorflow as tf
from dotenv import load_dotenv

# module imports
from trainer import Trainer
from transforms import ImageTransform
from architecture import ResNetArchitecture, GANArchitecture
from model import SuperResolutionResNet, Generator, Discriminator, TruncatedVGG19

load_dotenv()

# Data parameters
data_folder = "./"  # folder with JSON data files
crop_size = 96  # crop size of target HR images
scaling_factor = 4  # the input LR images will be down-sampled from the target HR images by this factor

# Common Model parameters
large_kernel_size = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size = 3  # kernel size of the first and last convolutions which transform the inputs and outputs
n_channels = 128  # number of channels in-between, input and output channels for residual & subpixel conv blocks
n_blocks = 64  # number of residual blocks
srresnet_checkpoint = ""  # trained SRResNet checkpoint used for generator initialization

# Discriminator parameters
kernel_size_d = 3  # kernel size in all convolutional blocks
n_channels_d = 128  # number of channels in-between, input and output channels for residual & subpixel conv blocks
n_blocks_d = 8  # number of convolutional blocks
fc_size_d = 1024  # size of the first fully connected layer

# VGG parameters
vgg19_i = 5  # the index i in the definition for VGG loss; see paper or models.py
vgg19_j = 4  # the index j in the definition for VGG loss; see paper or models.py

# Learning parameters
batch_size = 1  # batch size
start_epoch = 0  # start at this epoch
epochs = 10_000  # number of training epochs
print_freq = 500  # print training status once every __ batches
lr = 1e-6  # learning rate
beta = 1e-3  # the coefficient to weight the adversarial loss in the perceptual loss


def main(architecture_type: str = "resnet"):
    """
    Manages the whole training pipeline given a model architecture.

    :param architecture_type: resnet or gan
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.MeanSquaredError()

    if architecture_type == "resnet":
        model = SuperResolutionResNet(
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size,
            n_channels=n_channels,
            n_blocks=n_blocks,
            scaling_factor=scaling_factor,
        )
        architecture = ResNetArchitecture(
            model=model, optimizer=optimizer, loss_fn=loss_fn
        )

    elif architecture_type == "gan":
        generator = Generator(
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size,
            n_channels=n_channels,
            n_blocks=n_blocks,
            scaling_factor=scaling_factor,
        )

        generator.initialize_with_srresnet(srresnet_checkpoint=srresnet_checkpoint)

        discriminator = Discriminator(
            kernel_size=kernel_size_d,
            n_channels=n_channels_d,
            n_blocks=n_blocks_d,
            fc_size=fc_size_d,
        )

        adversarial_loss = tf.keras.losses.BinaryCrossentropy()

        optimizer_d = tf.keras.optimizers.Adam(learning_rate=lr)

        transform = ImageTransform(
            split="train",
            crop_size=crop_size,
            lr_img_type="imagenet-norm",
            hr_img_type="[-1, 1]",
            scaling_factor=scaling_factor,
        )

        truncated_vgg19 = TruncatedVGG19(i=vgg19_j, j=vgg19_j)

        architecture = GANArchitecture(
            gen_model=generator,
            dis_model=discriminator,
            gen_optimizer=optimizer,
            dis_optimizer=optimizer_d,
            content_loss=loss_fn,
            adversarial_loss=adversarial_loss,
            transform=transform,
            vgg=truncated_vgg19,
        )
    else:
        raise NotImplementedError("Model architecture not implemented")

    trainer = Trainer(architecture=architecture, data_folder=data_folder)

    trainer.train(
        start_epoch=start_epoch,
        epochs=epochs,
        batch_size=batch_size,
        print_freq=print_freq,
    )


if __name__ == "__main__":
    main(architecture_type="resnet")
