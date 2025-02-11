import pytest
import torch
import pandas as pd
import os
from deepmaize.dataset import MaizeLeafDataset, get_transforms
from PIL import Image
import numpy as np

@pytest.fixture
def sample_image(tmp_path):
    # Create a sample image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "test_image.jpg"
    
    # Create a random image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(img_path)
    
    return img_dir

@pytest.fixture
def sample_csv(tmp_path):
    # Create a temporary CSV file
    df = pd.DataFrame({
        'Image': ['test_image'],
        'Label': [0]
    })
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def test_transforms():
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    # Create dummy image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Test transforms
    train_img = train_transform(img)
    val_img = val_transform(img)
    
    assert isinstance(train_img, torch.Tensor)
    assert isinstance(val_img, torch.Tensor)
    assert train_img.shape == (3, 224, 224)
    assert val_img.shape == (3, 224, 224)