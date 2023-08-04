# third-party imports
import tensorflow as tf
from pprint import PrettyPrinter

# module imports
from evaluator import Evaluator

# Instantiate pretty printer
pp = PrettyPrinter(indent=4)

# Test data folder and name
data_folder = "./"
test_data_name = "dummy"

# Image to resolve
image = "bird.jpeg"

if __name__ == "__main__":
    # Load models
    resnet = tf.saved_model.load("SuperResolutionResNet_99999")
    generator = tf.saved_model.load("Generator_99999")

    # Create evaluator
    evaluator = Evaluator(
        resnet=resnet,
        generator=generator,
        data_folder=data_folder,
        test_data_name=test_data_name,
    )

    # Evaluate models
    evaluator.evaluate()

    # Perform super resolution
    evaluator.super_resolve(img=image)

    # Print metrics
    print("Evaluation results:")
    pp.pprint(
        {
            "PSNR (SRResNet)": evaluator.PSNRs_resnet.result().numpy(),
            "PSNR (SRGAN)": evaluator.PSNRs_gan.result().numpy(),
            "SSIM (SRResNet)": evaluator.SSIMs_resnet.result().numpy(),
            "SSIM (SRGAN)": evaluator.SSIMs_gan.result().numpy(),
        }
    )
