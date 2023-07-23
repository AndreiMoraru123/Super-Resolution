# standard imports
from dotenv import load_dotenv

# third-party imports
import tensorflow as tf  # type: ignore
from tensorflow.keras.losses import Loss  # type:ignore
from PIL import Image  # type: ignore

# module imports
from dataset import create_dataset
from model import SuperResolutionResNet

load_dotenv()

# Data parameters
data_folder = './'  # folder with JSON data files
crop_size = 96  # crop size of target HR images
scaling_factor = 4  # the input LR images will be downsampled from the target HR images by this factor

# Model parameters
large_kernel_size = 9  # kernel size of the first and last convolutions which transform the inputs and outputs

small_kernel_size = 3  # kernel size of the first and last convolutions which transform the inputs and outputs
n_channels = 64  # number of channels in-between, input and output channels for residual & subpixel conv blocks
n_blocks = 16  # number of residual blocks

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 16  # batch size
start_epoch = 0  # start at this epoch
epochs = 50  # number of training epochs
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 500  # print training status once every __ batches
lr = 1e-5  # learning rate

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn = tf.keras.losses.MeanSquaredError()


@tf.function
def train_step(low_res_images, high_res_images, model, optimizer, loss_fn) -> Loss:
    """

    :param low_res_images: low resolution input images
    :param high_res_images: high resolution input images
    :param model: the Super Resolution ResNet model
    :param optimizer: the optimizer
    :param loss_fn: the mse loss
    :return:
    """
    with tf.GradientTape() as tape:
        sr_images = model(low_res_images, training=True)
        loss = loss_fn(high_res_images, sr_images)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def main():
    model = SuperResolutionResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                                  n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)

    model.compile(optimizer=optimizer, loss=loss_fn)

    train_dataset = create_dataset(data_folder=data_folder, split='train',
                                   crop_size=crop_size, scaling_factor=scaling_factor,
                                   lr_img_type='imagenet-norm', hr_img_type='[-1, 1]')
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    for epoch in range(start_epoch, epochs):
        for i, (lr_imgs, hr_imgs) in enumerate(train_dataset):
            lr_imgs = tf.dtypes.cast(lr_imgs, tf.float32)
            hr_imgs = tf.dtypes.cast(hr_imgs, tf.float32)

            loss = train_step(low_res_images=lr_imgs, high_res_images=hr_imgs,
                              model=model, optimizer=optimizer,
                              loss_fn=loss_fn)

            if i % print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{epochs}]----'
                      f'Loss {loss:.4f}')
    model.save('srresnet')


if __name__ == "__main__":
    main()
