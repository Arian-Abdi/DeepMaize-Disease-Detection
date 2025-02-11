import torch
import pytest
from deepmaize.model import SwinMaizeClassifier

def test_model_initialization():
    model = SwinMaizeClassifier(num_classes=4)
    assert isinstance(model, SwinMaizeClassifier)

def test_model_output_shape():
    model = SwinMaizeClassifier(num_classes=4)
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    output = model(x)
    assert output.shape == (batch_size, 4)

def test_model_forward_pass():
    model = SwinMaizeClassifier(num_classes=4)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    assert torch.is_tensor(output)
    assert not torch.isnan(output).any()