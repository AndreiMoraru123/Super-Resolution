# standard imports
import os
import json

# third-party imports
import tensorflow as tf  # type: ignore
from PIL import Image  # type: ignore

# module imports
from transforms import ImageTransform


def create_dataset(
    data_folder: str,
    split: str,
    crop_size: int,
    scaling_factor: int,
    lr_img_type: str,
    hr_img_type: str,
    test_data_name: str = '',
) -> tf.data.Dataset:
    """
    Create a Super Resolution (SR) dataset using TensorFlow's data API.

    :param data_folder: folder with JSON data files
    :param split: one of 'train' or 'test'
    :param crop_size: crop size of target HR images
    :param scaling_factor: the input LR images will be downsampled from the target HR images by this factor; the scaling done in the super-resolution
    :param lr_img_type: the format for the LR image supplied to the model; see convert_image() in utils.py for available formats
    :param hr_img_type: the format for the HR image supplied to the model; see convert_image() in utils.py for available formats
    :param test_data_name: if this is the 'test' split, which test dataset? (for example, "Set14")
    """
    assert split in {'train', 'test'}
    if split == 'test' and not test_data_name:
        raise ValueError("Please provide the name of the test dataset!")
    assert lr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
    assert hr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

    if split == 'train':
        with open(os.path.join(data_folder, 'train_images.json'), 'r') as f:
            images = json.load(f)
    else:
        with open(os.path.join(data_folder, test_data_name + '_test_images.json'), 'r') as f:
            images = json.load(f)

    transform = ImageTransform(split=split,
                               crop_size=crop_size,
                               lr_img_type=lr_img_type,
                               hr_img_type=hr_img_type,
                               scaling_factor=scaling_factor)

    def generator():
        """Data generator for the TensorFlow Dataset."""
        for image_path in images:
            img = Image.open(image_path, mode='r')
            img = img.convert('RGB')
            # Transform
            lr_img, hr_img = transform(img)
            # Generate
            yield lr_img, hr_img

    return tf.data.Dataset.from_generator(generator=generator, output_signature=(
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32)))
