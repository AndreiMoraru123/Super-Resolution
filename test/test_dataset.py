# standard imports
import json
from unittest.mock import patch, MagicMock

# third party imports
import pytest  # type: ignore
import tensorflow as tf  # type: ignore
from PIL import Image  # type: ignore

# module imports
from trainer import Trainer


@pytest.fixture(name="json_path")
def json_data(tmpdir) -> str:
    """Temporary JSON data folders."""

    json_data = ["image1.jpg", "image2.jpg", "image3.jpg"]
    train_json = tmpdir.join("train_images.json")
    test_json = tmpdir.join("dummy_test_images.json")

    with open(train_json, "w") as j:
        json.dump(json_data, j)

    with open(test_json, "w") as j:
        json.dump(json_data, j)

    return str(tmpdir)


@pytest.fixture(name="config")
def dataset_config(json_path):
    """Configuration parameters for creating the Dataset."""

    config = {
        "data_folder": json_path,
        "split": "train",
        "crop_size": 96,
        "scaling_factor": 4,
        "low_res_img_type": "[0, 255]",
        "high_res_img_type": "[0, 255]",
        "test_data_name": "dummy",
    }
    return config


@pytest.fixture(name="trainer")
def mock_trainer(config):
    """Mock Trainer with no compilation step and mock architecture."""
    mock_compile = patch.object(Trainer, "compile", return_value=None)
    mock_compile.start()

    instance = Trainer(data_folder=config["data_folder"], architecture=MagicMock())
    yield instance
    # Optional here, since I don't care if it persists
    mock_compile.stop()


@patch("PIL.Image.open")
def test_dataset_creation(mock_img_open, trainer, config):
    """Test Dataset creation with mocked paths and image."""

    # Mock image object
    mock_img = Image.new(mode="RGB", size=(150, 150))
    # Cannot crop a larger patch from a smaller image
    assert mock_img.size[0] > config["crop_size"]
    assert mock_img.size[1] > config["crop_size"]
    # Mock RGB conversion
    mock_img.convert = MagicMock(return_value=mock_img)
    # Return the mock image on PIL Image open
    mock_img_open.return_value = mock_img
    # Create Dataset from config
    dataset = trainer.create_dataset(**config)
    # "image1.jpg", "image2.jpg", "image3.jpg"
    assert len(list(dataset.as_numpy_iterator())) == 3
    # assert data content
    for lr_img, hr_img in dataset:
        # Check dims
        assert tf.rank(lr_img) == 3
        assert tf.rank(hr_img) == 3
        # Check color channels
        assert lr_img.shape[-1] == 3
        assert hr_img.shape[-1] == 3
        # Check data type
        assert lr_img.dtype == tf.float32
        assert hr_img.dtype == tf.float32
        # Assert transform from high to low resolution took place
        assert hr_img.shape[0] // config["scaling_factor"] == lr_img.shape[0]
        assert hr_img.shape[1] // config["scaling_factor"] == lr_img.shape[1]
