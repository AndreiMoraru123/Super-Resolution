import os
import pytest
import tensorflow as tf  # type: ignore


@pytest.fixture(autouse=True)
def use_cpu_only():
    """Conv2D and tf.nn.Conv2D are broken on my CUDA/cuDNN version for some reason."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.config.set_visible_devices([], 'GPU')
    yield
