# third-party imports
import tensorflow as tf  # type: ignore
from tensorflow.keras.losses import Loss  # type:ignore
from PIL import Image  # type: ignore

# module imports
from dataset import create_dataset
from transforms import ImageTransform
from model import Generator, Discriminator, TruncatedVGG19

# Data parameters
data_folder = './'  # folder with JSON data files
crop_size = 96  # crop size of target HR images
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

# Generator parameters
large_kernel_size_g = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size_g = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
n_channels_g = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks_g = 16  # number of residual blocks
srresnet_checkpoint = "srresnet"  # filepath of the trained SRResNet checkpoint used for initialization

# Discriminator parameters
kernel_size_d = 3  # kernel size in all convolutional blocks
n_channels_d = 64  # number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
n_blocks_d = 8  # number of convolutional blocks
fc_size_d = 1024  # size of the first fully connected layer

# Learning parameters
checkpoint = None  # path to model (SRGAN) checkpoint, None if none
batch_size = 16  # batch size
start_epoch = 0  # start at this epoch
epochs = 50  # number of training epochs
workers = 4  # number of workers for loading data in the DataLoader
vgg19_i = 5  # the index i in the definition for VGG loss; see paper or models.py
vgg19_j = 4  # the index j in the definition for VGG loss; see paper or models.py
beta = 1e-3  # the coefficient to weight the adversarial loss in the perceptual loss
print_freq = 500  # print training status once every __ batches
lr = 1e-4  # learning rate

optimizer_g = tf.keras.optimizers.Adam(learning_rate=lr)
optimizer_d = tf.keras.optimizers.Adam(learning_rate=lr)
content_loss = tf.keras.losses.MeanSquaredError()
adversarial_loss = tf.keras.losses.BinaryCrossentropy()

tf.config.set_visible_devices([], 'GPU')


def train_step(low_res_images, high_res_images, generator, discriminator, adversarial_loss,
               truncated_vgg, optimizer_d, optimizer_g, content_loss, transform) -> Loss:
    """

    :param low_res_images:
    :param high_res_images:
    :param generator:
    :param discriminator:
    :param optimizer:
    :param content_loss:
    :param loss_fn:
    :return:
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        sr_images = generator(low_res_images)
        sr_images = transform.convert_image(sr_images, source='[-1, 1]', target='imagenet-norm')

        sr_images_in_vgg_space = truncated_vgg(sr_images)
        hr_images_in_vgg_space = truncated_vgg(high_res_images)

        sr_discriminated = discriminator(sr_images)

        c_loss = content_loss(sr_images_in_vgg_space, hr_images_in_vgg_space)
        a_loss = adversarial_loss(sr_discriminated, tf.ones_like(sr_discriminated))
        perceptual_loss = c_loss + beta * a_loss

        hr_discriminated = discriminator(high_res_images)
        disc_loss = adversarial_loss(sr_discriminated, tf.zeros_like(sr_discriminated)) + \
                    adversarial_loss(hr_discriminated, tf.ones_like(hr_discriminated))

    generator_gradients = gen_tape.gradient(perceptual_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    optimizer_g.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    optimizer_d.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return perceptual_loss, disc_loss


def main():
    generator = Generator(large_kernel_size=large_kernel_size_g,
                          small_kernel_size=small_kernel_size_g,
                          scaling_factor=scaling_factor,
                          n_channels=n_channels_g,
                          n_blocks=n_blocks_g)

    generator.initialize_with_srresnet(srresnet_checkpoint=srresnet_checkpoint)

    discriminator = Discriminator(kernel_size=kernel_size_d,
                                  n_channels=n_channels_d,
                                  n_blocks=n_blocks_d,
                                  fc_size=fc_size_d)

    truncated_vgg19 = TruncatedVGG19(i=vgg19_j, j=vgg19_j)

    train_dataset = create_dataset(data_folder=data_folder, split='train',
                                   crop_size=crop_size, scaling_factor=scaling_factor,
                                   lr_img_type='imagenet-norm', hr_img_type='[-1, 1]')

    transform = ImageTransform(split="train",
                               crop_size=crop_size,
                               lr_img_type='imagenet-norm',
                               hr_img_type='[-1, 1]',
                               scaling_factor=scaling_factor)

    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    for epoch in range(start_epoch, epochs):
        for i, (lr_imgs, hr_imgs) in enumerate(train_dataset):
            lr_imgs = tf.dtypes.cast(lr_imgs, tf.float32)
            hr_imgs = tf.dtypes.cast(hr_imgs, tf.float32)

            perceptual_loss, discriminator_loss = train_step(low_res_images=lr_imgs, high_res_images=hr_imgs,
                                                             discriminator=discriminator, generator=generator,
                                                             truncated_vgg=truncated_vgg19,
                                                             content_loss=content_loss,
                                                             adversarial_loss=adversarial_loss,
                                                             optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                                                             transform=transform)

            if i % print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{epochs}]----'
                      f'Perceptual Loss {perceptual_loss:.4f}'
                      f'Discriminator Loss {discriminator_loss:.4f}')
    generator.save('srgenerator')
    discriminator.save('srdiscriminator')


if __name__ == "__main__":
    main()
