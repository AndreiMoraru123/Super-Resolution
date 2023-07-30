# standard imports
import random
from typing import Tuple, Union

# third-party imports
import numpy as np
import tensorflow as tf  # type: ignore
from PIL import Image  # type: ignore


class ImageTransform(object):
    """
    Image transformation pipeline.
    """

    def __init__(
        self,
        split: str,
        crop_size: int,
        scaling_factor: int,
        lr_img_type: str,
        hr_img_type: str,
    ):
        """
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of HR images
        :param scaling_factor: LR images will be down-sampled from the HR images by this factor
        :param lr_img_type: the target format for the LR image; see convert_image method below for available formats
        :param hr_img_type: the target format for the HR image; see convert_image method below for available formats
        """

        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.rgb_weights = tf.constant([65.481, 128.553, 24.966], dtype=tf.float32)

        # CHW
        self.imagenet_mean_cpu = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)[
            None, None, :
        ]
        self.imagenet_std_cpu = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)[
            None, None, :
        ]
        # NHWC
        self.imagenet_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)[
            None, None, None, :
        ]
        self.imagenet_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)[
            None, None, None, :
        ]

        assert self.split in {"train", "test"}

    def convert_image(
        self, img: Union[Image.Image, tf.Tensor], source: str, target: str
    ) -> Union[Image.Image, tf.Tensor]:
        """
        Convert an image from a source format to a target format.

        :param img: image
        :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
        :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                       'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                       'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
        :return: converted image
        """

        assert source in {"pil", "[0, 1]", "[-1, 1]"}, (
            "Cannot convert from source format %s!" % source
        )
        assert target in {
            "pil",
            "[0, 255]",
            "[0, 1]",
            "[-1, 1]",
            "imagenet-norm",
            "y-channel",
        }, (
            "Cannot convert to target format %s!" % target
        )

        # Convert from source to [0, 1]
        if source == "pil":
            img = tf.convert_to_tensor(np.array(img), dtype=tf.float32) / 255.0
        elif source == "[0, 1]":
            pass
        elif source == "[-1, 1]":
            img = (img + 1.0) / 2.0

        # Convert from [0, 1] to target
        if target == "pil":
            img = tf.keras.preprocessing.image.array_to_img(img)
        elif target == "[0, 255]":
            img = 255.0 * img
        elif target == "[0, 1]":
            pass  # already in [0, 1]
        elif target == "[-1, 1]":
            img = 2.0 * img - 1
        elif target == "imagenet-norm":
            if len(img.shape) == 3:
                img = (img - self.imagenet_mean_cpu) / self.imagenet_std_cpu
            elif len(img.shape) == 4:
                img = (img - self.imagenet_mean) / self.imagenet_std
        elif target == "y-channel":
            img = (
                tf.tensordot(
                    img[:, 4:-4, 4:-4, :] * 255, self.rgb_weights, axes=[[3], [0]]
                )
                / 255.0
                + 16.0
            )

        return img

    def __call__(
        self, img: Image.Image
    ) -> Tuple[Union[Image.Image, tf.Tensor], Union[Image.Image, tf.Tensor]]:
        """
        :param img: a PIL source image from which the HR image will be cropped + down-sampled to create the LR image
        :return: LR and HR images in the specified format
        """

        # Crop
        if self.split == "train":
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # Largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # Downsize this crop to obtain a low-resolution version of it
        lr_img = hr_img.resize(
            (
                int(hr_img.width / self.scaling_factor),
                int(hr_img.height / self.scaling_factor),
            ),
            Image.BICUBIC,
        )

        # Sanity check
        assert (hr_img.width == lr_img.width * self.scaling_factor) and (
            hr_img.height == lr_img.height * self.scaling_factor
        )

        # Convert the LR and HR image to the required type
        lr_img = self.convert_image(lr_img, source="pil", target=self.lr_img_type)
        hr_img = self.convert_image(hr_img, source="pil", target=self.hr_img_type)

        return lr_img, hr_img
