# standard imports
import os
import json

# third-party imports
import tensorflow as tf
from tensorflow.keras import Model  # type: ignore
from tensorflow.keras.metrics import Mean  # type: ignore
from PIL import Image, ImageDraw, ImageFont  # type: ignore
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# module imports
from transforms import ImageTransform


class Evaluator:
    """Utility class to evaluate super resolution models."""

    def __init__(
        self,
        resnet: Model,
        generator: Model,
        data_folder: str,
        crop_size: int = 96,
        scaling_factor: int = 4,
        low_res_image_type: str = "imagenet-norm",
        high_res_image_type: str = "[-1, 1]",
        test_data_name: str = "dummy",
    ):
        """
        :param resnet: the SRResNet TF model to be evaluated
        :param generator: the Generator (SRGAN) TF model to be evaluated
        :param data_folder: folder in which the test data is stored
        :param crop_size: cropping size for transforms during training
        :param scaling_factor: up-scaling factor for higher resolution
        :param low_res_image_type: low resolution image type for transform
        :param high_res_image_type: high resolution image type for transform
        :param test_data_name: json file(s) with images names for the test set
        """
        self.resnet_inference = resnet.signatures["serving_default"]
        self.generator_inference = generator.signatures["serving_default"]

        self.dataset = self.create_dataset(
            data_folder=data_folder,
            crop_size=crop_size,
            high_res_img_type=high_res_image_type,
            low_res_img_type=low_res_image_type,
            scaling_factor=scaling_factor,
            test_data_name=test_data_name,
            split="test",
        )

        self.transform = ImageTransform(
            split="test",
            crop_size=crop_size,
            lr_img_type="imagenet-norm",
            hr_img_type="[-1, 1]",
            scaling_factor=scaling_factor,
        )

        self.PSNRs_resnet = Mean()
        self.PSNRs_gan = Mean()
        self.SSIMs_resnet = Mean()
        self.SSIMs_gan = Mean()

    def evaluate(self):
        """evaluates the model using peak signal-to-noise ratio and structural similarity."""

        for _, (low_res_images, high_res_images) in enumerate(self.dataset):
            super_res_images_resnet = self.resnet_inference(
                tf.expand_dims(low_res_images, axis=0)
            )["output_0"]
            super_res_images_srgan = self.generator_inference(
                tf.expand_dims(low_res_images, axis=0)
            )["output_0"]

            super_res_images_resnet_y = self.transform.convert_image(
                super_res_images_resnet, source="[-1, 1]", target="y-channel"
            )
            super_res_images_srgan_y = self.transform.convert_image(
                super_res_images_srgan, source="[-1, 1]", target="y-channel"
            )

            super_res_images_resnet_y = tf.squeeze(super_res_images_resnet_y, axis=0)
            super_res_images_srgan_y = tf.squeeze(super_res_images_srgan_y, axis=0)

            high_res_images_y = self.transform.convert_image(
                tf.expand_dims(high_res_images, axis=0),
                source="[-1, 1]",
                target="y-channel",
            )
            high_res_images_y = tf.squeeze(high_res_images_y, axis=0)

            psnr_resnet = peak_signal_noise_ratio(
                high_res_images_y.numpy(),
                super_res_images_resnet_y.numpy(),
                data_range=255.0,
            )
            psnr_srgan = peak_signal_noise_ratio(
                high_res_images_y.numpy(),
                super_res_images_srgan_y.numpy(),
                data_range=255.0,
            )

            ssim_resnet = structural_similarity(
                high_res_images_y.numpy(),
                super_res_images_resnet_y.numpy(),
                data_range=255.0,
            )
            ssim_srgan = structural_similarity(
                high_res_images_y.numpy(),
                super_res_images_srgan_y.numpy(),
                data_range=255.0,
            )

            self.PSNRs_resnet.update_state(psnr_resnet)
            self.PSNRs_gan.update_state(psnr_srgan)

            self.SSIMs_resnet.update_state(ssim_resnet)
            self.SSIMs_gan.update_state(ssim_srgan)

    def super_resolve(self, img: str, halve: bool = False):
        """Adds super resolution method with both models to the class."""

        # Load image, down-sample to obtain low-res version
        hr_img = Image.open(img, mode="r")
        hr_img = hr_img.convert("RGB")

        if halve:
            hr_img = hr_img.resize(
                (int(hr_img.width / 2), int(hr_img.height / 2)), Image.LANCZOS
            )

        # Create low resolution image at runtime
        lr_img = hr_img.resize(
            (int(hr_img.width / 4), int(hr_img.height / 4)), Image.BICUBIC
        )

        # Bicubic Up-sampling
        bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

        lr_img = tf.expand_dims(
            self.transform.convert_image(lr_img, source="pil", target="imagenet-norm"),
            axis=0,
        )

        # Super-resolution (SR) with SRResNet
        sr_img_srresnet = self.resnet_inference(lr_img)
        sr_img_srresnet = tf.squeeze(sr_img_srresnet["output_0"])
        sr_img_srresnet = self.transform.convert_image(
            sr_img_srresnet, source="[-1, 1]", target="pil"
        )

        # Super-resolution (SR) with SRGAN
        sr_img_srgan = self.generator_inference(lr_img)
        sr_img_srgan = tf.squeeze(sr_img_srgan["output_0"])
        sr_img_srgan = self.transform.convert_image(
            sr_img_srgan, source="[-1, 1]", target="pil"
        )

        # Create grid
        margin = 40
        grid_img = Image.new(
            "RGB",
            (2 * hr_img.width + 3 * margin, 2 * hr_img.height + 3 * margin),
            (255, 255, 255),
        )

        # Drawer and font
        draw = ImageDraw.Draw(grid_img)
        font = ImageFont.load_default()

        # Place bicubic-upsampled image
        grid_img.paste(bicubic_img, (margin, margin))
        draw.text(
            (margin + bicubic_img.width / 2, margin - 10),
            "Bicubic",
            font=font,
            fill="black",
        )

        # Place SRResNet image
        grid_img.paste(sr_img_srresnet, (2 * margin + bicubic_img.width, margin))
        draw.text(
            (2 * margin + bicubic_img.width + sr_img_srresnet.width / 2, margin - 10),
            "SRResNet",
            font=font,
            fill="black",
        )

        # Place SRGAN image
        grid_img.paste(sr_img_srgan, (margin, 2 * margin + sr_img_srresnet.height))
        draw.text(
            (margin + bicubic_img.width / 2, 2 * margin + sr_img_srresnet.height - 10),
            "SRGAN",
            font=font,
            fill="black",
        )

        # Place original HR image
        grid_img.paste(
            hr_img,
            (2 * margin + bicubic_img.width, 2 * margin + sr_img_srresnet.height),
        )
        draw.text(
            (
                2 * margin + bicubic_img.width + sr_img_srresnet.width / 2,
                2 * margin + sr_img_srresnet.height - 10,
            ),
            "Original HR",
            font=font,
            fill="black",
        )

        # Save image
        grid_img.save(img[:-5] + "_resolved" + ".png")

    @staticmethod
    def create_dataset(
        data_folder: str,
        split: str,
        crop_size: int,
        scaling_factor: int,
        low_res_img_type: str,
        high_res_img_type: str,
        test_data_name: str = "",
    ) -> tf.data.Dataset:
        """
        Create a Super Resolution (SR) dataset using TensorFlow's data API.

        :param data_folder: folder with JSON data files
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of target HR images
        :param scaling_factor: the input LR images will be down-sampled from the target HR images by this factor
        :param low_res_img_type: the format for the LR image supplied to the model
        :param high_res_img_type: the format for the HR image supplied to the model
        :param test_data_name: if this is the 'test' split, which test dataset? (for example, "Set14")
        """
        assert split == "test"

        if not test_data_name:
            raise ValueError("Please provide the name of the test dataset!")

        assert low_res_img_type in {"[0, 255]", "[0, 1]", "[-1, 1]", "imagenet-norm"}
        assert high_res_img_type in {"[0, 255]", "[0, 1]", "[-1, 1]", "imagenet-norm"}

        with open(
            os.path.join(data_folder, test_data_name + "_test_images.json"), "r"
        ) as f:
            images = json.load(f)

        transform = ImageTransform(
            split=split,
            crop_size=crop_size,
            lr_img_type=low_res_img_type,
            hr_img_type=high_res_img_type,
            scaling_factor=scaling_factor,
        )

        def generator():
            """Data generator for the TensorFlow Dataset."""

            for image_path in images:
                img = Image.open(image_path, mode="r")
                img = img.convert("RGB")
                # Transform
                lr_img, hr_img = transform(img)
                # Generate
                yield lr_img, hr_img

        return tf.data.Dataset.from_generator(
            generator=generator,
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            ),
        )
