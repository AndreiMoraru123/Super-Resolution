# third-party imports
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# module imports
from transforms import ImageTransform


crop_size = 96  # crop size of target HR images
scaling_factor = 4  # the input LR images will be down-sampled from the target HR images by this factor

resnet = tf.saved_model.load("SuperResolutionResNet_9999")
resnet_inference = resnet.signatures["serving_default"]

generator = tf.saved_model.load("Generator_9999")
generator_inference = generator.signatures["serving_default"]

# Need this instance for conversions
transform = ImageTransform(split="train",
                           crop_size=crop_size,
                           lr_img_type='imagenet-norm',
                           hr_img_type='[-1, 1]',
                           scaling_factor=scaling_factor)


def super_resolve(img: str, halve: bool = False):
    """
    Visualizes the super-resolved images from the SRResNet and SRGAN for comparison with the bicubic up-sampled image
    and the original high-resolution (HR) image, as done in the paper.

    :param img: filepath of the HR image
    :param halve: halve each dimension of the HR image to make sure it's not greater than the dimensions of your screen?
                  For instance, for a 2160p HR image, the LR image will be of 540p (1080p/4) resolution. On a 1080p
                  screen, you will therefore be looking at a comparison between a 540p LR image and a 1080p SR/HR image
                  because your 1080p screen can only display the 2160p SR/HR image at a down-sampled 1080p. This is only
                  an APPARENT rescaling of 2x.
                  If you want to reduce HR resolution by a different extent, modify accordingly.
    """

    # Load image, down-sample to obtain low-res version
    hr_img = Image.open(img, mode="r")
    hr_img = hr_img.convert('RGB')

    if halve:
        hr_img = hr_img.resize((int(hr_img.width / 2), int(hr_img.height / 2)),
                               Image.LANCZOS)

    lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)),
                           Image.BICUBIC)

    # Bicubic Upsampling
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    lr_img = tf.expand_dims(transform.convert_image(lr_img, source='pil', target='imagenet-norm'), axis=0)

    # Super-resolution (SR) with SRResNet
    sr_img_srresnet = resnet_inference(lr_img)
    sr_img_srresnet = tf.squeeze(sr_img_srresnet['output_0'])
    sr_img_srresnet = transform.convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')

    sr_img_srgan = generator_inference(lr_img)
    sr_img_srgan = tf.squeeze(sr_img_srgan['output_0'])
    sr_img_srgan = transform.convert_image(sr_img_srgan, source='[-1, 1]', target='pil')

    # Create grid
    margin = 40
    grid_img = Image.new('RGB', (2 * hr_img.width + 3 * margin, 2 * hr_img.height + 3 * margin), (255, 255, 255))

    # Drawer and font
    draw = ImageDraw.Draw(grid_img)
    font = ImageFont.load_default()

    # Place bicubic-upsampled image
    grid_img.paste(bicubic_img, (margin, margin))
    text_size = font.getbbox("Bicubic")
    draw.text(xy=[margin + bicubic_img.width / 2 - text_size[0] / 2, margin - text_size[1] - 5], text="Bicubic",
              font=font,
              fill='black')

    # Place SRResNet image
    grid_img.paste(sr_img_srresnet, (2 * margin + bicubic_img.width, margin))
    text_size = font.getbbox("SRResNet")
    draw.text(
        xy=[2 * margin + bicubic_img.width + sr_img_srresnet.width / 2 - text_size[0] / 2, margin - text_size[1] - 5],
        text="SRResNet", font=font, fill='black')

    # Place SRGAN image
    grid_img.paste(sr_img_srgan, (margin, 2 * margin + sr_img_srresnet.height))
    text_size = font.getbbox("SRGAN")
    draw.text(
        xy=[margin + bicubic_img.width / 2 - text_size[0] / 2, 2 * margin + sr_img_srresnet.height - text_size[1] - 5],
        text="SRGAN", font=font, fill='black')

    # Place original HR image
    grid_img.paste(hr_img, (2 * margin + bicubic_img.width, 2 * margin + sr_img_srresnet.height))
    text_size = font.getbbox("Original HR")
    draw.text(xy=[2 * margin + bicubic_img.width + sr_img_srresnet.width / 2 - text_size[0] / 2,
                  2 * margin + sr_img_srresnet.height - text_size[1] - 1], text="Original HR", font=font, fill='black')

    # Save image
    grid_img.save(img[:-5] + "_resolved" + ".png")


if __name__ == '__main__':
    super_resolve("bird.jpeg", halve=False)
