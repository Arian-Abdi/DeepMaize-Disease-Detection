import pytest
import torch
from deepmaize.predict import predict_image, predict_batch
from deepmaize.model import SwinMaizeClassifier
from PIL import Image
import numpy as np

@pytest.fixture
def model():
    return SwinMaizeClassifier(num_classes=4)

@pytest.fixture
def sample_image(tmp_path):
    # Create a sample image
    img_path = tmp_path / "test_image.jpg"
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(img_path)
    return img_path

def test_predict_output_format(model, sample_image):
    result = predict_image(model, sample_image, device='cpu')  # Use CPU for testing
    
    assert isinstance(result, dict)
    assert 'class' in result
    assert 'confidence' in result
    assert 'class_id' in result
    
    assert isinstance(result['confidence'], float)
    assert 0 <= result['confidence'] <= 1
    assert isinstance(result['class_id'], int)
    assert 0 <= result['class_id'] < 4

def test_predict_batch(model, sample_image):
    images = [sample_image] * 3  # Test with 3 copies of the same image
    results = predict_batch(model, images, device='cpu')  # Use CPU for testing
    
    assert isinstance(results, list)
    assert len(results) == 3
    
    for result in results:
        assert isinstance(result, dict)
        assert 'class' in result
        assert 'confidence' in result
        assert 'class_id' in result