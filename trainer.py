# standard imports
import os
import json
import shutil
import logging

# third-party imports
import tensorflow as tf  # type: ignore
from tensorflow.keras.metrics import Mean  # type: ignore
from tensorboard.plugins import projector  # type: ignore
from colorama import Fore, init  # type: ignore
from PIL import Image  # type: ignore

# module imports
from transforms import ImageTransform
from architecture import Architecture, ResNetArchitecture, GANArchitecture

# logging house-keeping
init(autoreset=True)
logging.basicConfig(level=logging.INFO)


class Trainer:
    """Utility class to train super resolution models."""

    def __init__(
        self,
        architecture: Architecture,
        data_folder: str,
        crop_size: int = 96,
        scaling_factor: int = 4,
        low_res_image_type: str = "imagenet-norm",
        high_res_image_type: str = "[-1, 1]",
        log_dir: str = "logs",
    ):
        """
        Initializes the trainer with the given architecture.

        :param architecture: Architecture (model + optimizer + loss)
        :param data_folder: folder in which the train data is stored
        :param crop_size: cropping size for transforms during training
        :param scaling_factor: up-scaling factor for higher resolution
        :param low_res_image_type: low resolution image type for transform
        :param high_res_image_type: high resolution image type for transform
        :param log_dir: directory location for TensorBoard logging
        """
        self.architecture = architecture
        self.data_folder = data_folder
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.low_res_image_type = low_res_image_type
        self.high_res_image_type = high_res_image_type
        self.log_dir = log_dir

        if os.path.exists(log_dir):
            logging.info(f"{Fore.YELLOW}Flushing Logs")
            shutil.rmtree(log_dir)

        logging.info(f"{Fore.CYAN}Creating Summary Writer")
        self.summary_writer = tf.summary.create_file_writer(log_dir)

        logging.info(f"{Fore.MAGENTA}Creating Dataset")
        self.dataset = self.create_dataset(
            data_folder=data_folder,
            crop_size=crop_size,
            high_res_img_type=high_res_image_type,
            low_res_img_type=low_res_image_type,
            scaling_factor=scaling_factor,
            split="train",
        )
        logging.info(f"{Fore.GREEN}Compiling Model")
        self.compile()

    def compile(self):
        """Compiles the model with the optimizer and loss criterion."""

        if isinstance(self.architecture, GANArchitecture):
            self.architecture.model.compile(
                optimizer=self.architecture.optimizer, loss=self.architecture.loss_fn
            )
            self.architecture.model2.compile(
                optimizer=self.architecture.optimizer2, loss=self.architecture.loss_fn2
            )
        elif isinstance(self.architecture, ResNetArchitecture):
            self.architecture.model.compile(
                optimizer=self.architecture.optimizer, loss=self.architecture.loss_fn
            )
        else:
            raise NotImplementedError(
                "Trainer not defined for this type of architecture"
            )

    def save_checkpoint(self, name: str, epoch: int):
        """
        Saves the model checkpoint at the given epoch.

        :param name: model name
        :param epoch: the given epoch for which to save the model.
        """

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)
            ]
        )
        def serving_fn(image_batch):
            """
            Serving function for saving the model.

            :param image_batch: input image place-holder
            :return: model inference function
            """
            return self.architecture.model(image_batch)

        tf.saved_model.save(
            self.architecture.model,
            export_dir=f"{name}_{epoch}/",
            signatures=serving_fn,
        )

    def train(self, start_epoch: int, epochs: int, batch_size: int, print_freq: int):
        """
        Train the given model architecture.

        :param start_epoch: starting epoch
        :param epochs: total number of epochs
        :param batch_size: how many images the model sees at once
        :param print_freq: log stats with this frequency
        """

        self.dataset = self.dataset.batch(batch_size=batch_size)
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)

        for epoch in range(start_epoch, epochs):
            for i, (low_res_images, high_res_imgs) in enumerate(self.dataset):
                low_res_images = tf.dtypes.cast(low_res_images, tf.float32)
                high_res_imgs = tf.dtypes.cast(high_res_imgs, tf.float32)

                loss = self.architecture.train_step(
                    low_res_images=low_res_images, high_res_images=high_res_imgs
                )

                with self.summary_writer.as_default():
                    if isinstance(loss, tuple):
                        gen_loss, dis_loss = loss
                        if i % print_freq == 0:
                            logging.info(
                                f"{Fore.GREEN}Epoch: [{epoch}][{i}/{epochs}]----"
                                f"{Fore.YELLOW}Generator Loss {gen_loss:.4f}----"
                                f"{Fore.CYAN}Discriminator Loss {dis_loss:.4f}"
                            )
                            tf.summary.scalar("Generator Loss", gen_loss, step=i)
                            tf.summary.scalar("Discriminator Loss", dis_loss, step=i)
                    else:
                        if i % print_freq == 0:
                            logging.info(
                                f"{Fore.GREEN}Epoch: [{epoch}][{i}/{epochs}]----"
                                f"{Fore.BLUE}Loss {loss:.4f}"
                            )
                            tf.summary.scalar("Loss", loss, step=i)

            if (epoch + 1) % 100_000 == 0:
                self.save_checkpoint(
                    name=self.architecture.model.__class__.__name__, epoch=epoch
                )

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
        assert split == "train"

        assert low_res_img_type in {"[0, 255]", "[0, 1]", "[-1, 1]", "imagenet-norm"}
        assert high_res_img_type in {"[0, 255]", "[0, 1]", "[-1, 1]", "imagenet-norm"}

        with open(os.path.join(data_folder, "train_images.json"), "r") as f:
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
