# Photo-Realistic Single Image Super-Resolution GAN

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)  ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)  ![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> [!NOTE]\
> I adapted the code from [this awesome PyTorch version](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution). Please check it out as well.

> [!IMPORTANT]\
> I am using python `3.11` with tensorflow `2.12` on WSL2.

## Steps
 1. `pip install -r requirements.txt`

## Overfitting on one image

![bird_resolved](https://github.com/AndreiMoraru123/Super-Resolution/assets/81184255/1429d7c7-96be-4737-be21-253ac5a09ed2)

```
Evaluation results:
{   'PSNR (SRGAN)': 29.44381,
    'PSNR (SRResNet)': 29.380556,
    'SSIM (SRGAN)': 0.87486875,
    'SSIM (SRResNet)': 0.87476}
```
