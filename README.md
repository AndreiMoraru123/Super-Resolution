# Photo-Realistic Single Image Super-Resolution GAN

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)  ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)  ![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> [!NOTE]\
> I adapted the code from [this awesome PyTorch version](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution). Please check it out as well.

> [!IMPORTANT]\
> I am using python `3.11` with tensorflow `2.12` on WSL2.

## Steps
 1. `pip install -r requirements.txt`
 2. download the [COCO dataset](https://cocodataset.org/#home) (I use COCO 2017).
 3. download the [Set5, Set14, BSD100 test datasets](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#common-image-sr-datasets)
 4. [create_data_lists.py](https://github.com/AndreiMoraru123/Super-Resolution/blob/main/create_data_lists.py) creates the JSON files for training and testing from the COCO image paths.
 5. [architecture.py](https://github.com/AndreiMoraru123/Super-Resolution/blob/main/architecture.py) defines the training blueprints for both models (ResNet and GAN).
 6. [train.py](https://github.com/AndreiMoraru123/Super-Resolution/blob/main/train.py) runs the whole training pipeline with top-down logic found in the file. Everything is managed by  the `Trainer` from [trainer.py](https://github.com/AndreiMoraru123/Super-Resolution/blob/main/trainer.py).
 7. [resolve.py](https://github.com/AndreiMoraru123/Super-Resolution/blob/main/resolve.py) generates the super resolution images from a given high resolution image (the low resolution version to be solved is generated by down-sampling the given image) and evaluates the models using with `scikit-image`'s `peak_signal_noise_ratio` and `structural_similarity` using the `Evaluator` from [evaluator.py](https://github.com/AndreiMoraru123/Super-Resolution/blob/main/evaluator.py).

The code itself is heavily commented and you can get a feel for super-resolution models by looking at the [tests](https://github.com/AndreiMoraru123/Neural-Machine-Translation/tree/main/test).

## Overfitting on one image

- **Top-Left**: Bicubic Up-sampling
- **Top-Right**: Super Resolution ResNet
- **Bottom-Left**: Super Resolution GAN
- **Bottom-Right**: Original High Resolution
  
![bird_resolved](https://github.com/AndreiMoraru123/Super-Resolution/assets/81184255/1429d7c7-96be-4737-be21-253ac5a09ed2)

```
Evaluation results:
{   'PSNR (SRGAN)': 29.44381,
    'PSNR (SRResNet)': 29.380556,
    'SSIM (SRGAN)': 0.87486875,
    'SSIM (SRResNet)': 0.87476}
```
