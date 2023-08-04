# standard imports
from typing import Optional, Tuple
from abc import ABC, abstractmethod

# third-party imports
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer
from PIL import Image

# module imports
from transforms import ImageTransform


class Architecture(ABC):
    """
    Model architecture template class
    """

    def __init__(
        self,
        model: Model,
        loss_fn: Loss,
        optimizer: Optimizer,
        model2: Optional[Model] = None,
        loss_fn2: Optional[Loss] = None,
        optimizer2: Optional[Optimizer] = None,
    ):
        """
        Initializes the model along with its corresponding loss and optimizer
        :param model: first model, mandatory
        :param loss_fn: the loss function of the first model, mandatory
        :param optimizer: the optimizer of the first model, mandatory
        :param model2: second model
        :param loss_fn2: the loss function of the second model
        :param optimizer2:  the optimizer of the second model
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.model2 = model2
        self.loss_fn2 = loss_fn2
        self.optimizer2 = optimizer2

    @abstractmethod
    def train_step(self, low_res_images: Image.Image, high_res_images: Image.Image):
        """
        Training step for model.

        :param low_res_images: low resolution input images
        :param high_res_images: high resolution target images
        """
        pass


class ResNetArchitecture(Architecture):
    """Super Resolution ResNet."""

    @tf.function(jit_compile=True)
    def train_step(
        self, low_res_images: Image.Image, high_res_images: Image.Image
    ) -> Loss:
        with tf.GradientTape() as tape:
            super_res_images = self.model(low_res_images, training=True)
            loss = self.loss_fn(high_res_images, super_res_images)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss


class GANArchitecture(Architecture):
    """Super Resolution GAN."""

    def __init__(
        self,
        gen_model: Model,
        dis_model: Model,
        gen_optimizer: Optimizer,
        dis_optimizer: Optimizer,
        content_loss: Loss,
        adversarial_loss: Loss,
        transform: ImageTransform,
        vgg: Model,
        beta: float = 1e-3,
    ):
        """
        Overrides the Architecture initialization with GAN specific parameters
        :param gen_model: generator
        :param dis_model: discriminator
        :param gen_optimizer: optimizer for generator
        :param dis_optimizer: optimizer for discriminator
        :param content_loss: the loss that when added to the adversarial loss becomes the loss for the generator
        :param adversarial_loss: the loss for the discriminator
        :param transform: image transform
        :param vgg: Optional truncated VGG19 to project the predictions into a dimension where the loss makes more sense
        :param beta: the coefficient to weight the adversarial loss in the perceptual loss
        """
        super().__init__(
            model=gen_model,
            loss_fn=content_loss,
            optimizer=gen_optimizer,
            model2=dis_model,
            loss_fn2=adversarial_loss,
            optimizer2=dis_optimizer,
        )
        self.transform = transform
        self.vgg = vgg
        self.beta = beta

    @tf.function(jit_compile=True)
    def train_step(
        self, low_res_images: Image.Image, high_res_images: Image.Image
    ) -> Tuple[Loss, Loss]:
        with tf.GradientTape() as gen_tape:
            super_res_images = self.model(low_res_images, training=True)
            super_res_images = self.transform.convert_image(
                super_res_images, source="[-1, 1]", target="imagenet-norm"
            )
            super_res_images_vgg_space = self.vgg(super_res_images)
            high_res_images_vgg_space = self.vgg(
                tf.stop_gradient(high_res_images)
            )  # does not get updated

            super_res_discriminated = self.model2(super_res_images, training=True)

            content_loss = self.loss_fn(
                super_res_images_vgg_space, high_res_images_vgg_space
            )
            adversarial_loss = self.loss_fn2(
                super_res_discriminated, tf.ones_like(super_res_discriminated)
            )
            perceptual_loss = content_loss + self.beta * adversarial_loss

        gen_gradients = gen_tape.gradient(
            perceptual_loss, self.model.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(gen_gradients, self.model.trainable_variables)
        )

        with tf.GradientTape() as dis_tape:
            super_res_discriminated = self.model2(
                tf.stop_gradient(super_res_images), training=True
            )
            high_res_discriminated = self.model2(high_res_images, training=True)

            adversarial_loss = self.loss_fn2(
                super_res_discriminated, tf.zeros_like(super_res_discriminated)
            ) + self.loss_fn2(
                high_res_discriminated, tf.ones_like(high_res_discriminated)
            )

        dis_gradients = dis_tape.gradient(
            adversarial_loss, self.model2.trainable_variables
        )
        self.optimizer2.apply_gradients(
            zip(dis_gradients, self.model2.trainable_variables)
        )

        return perceptual_loss, adversarial_loss
