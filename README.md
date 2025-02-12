# DeepMaize Disease Detection 

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/Arian-Abdi/DeepMaize-Disease-Detection/actions)

Deep learning-powered maize leaf disease detection using the Swin Transformer architecture. This model achieves high accuracy in classifying four distinct leaf conditions: Healthy, Common Rust, Blight, and Gray Leaf Spot.

## Key Features 

- **High Accuracy**: 97%+ accuracy across all disease classes
- **State-of-the-Art Architecture**: Utilizes Swin Transformer for superior feature extraction
- **Production Ready**: Complete with data preprocessing, training, and inference pipelines
- **Test Coverage**: Comprehensive test suite with all tests passing
- **Easy to Use**: Simple API for both training and prediction

## Model Performance ( F1 Score)

| Disease Class    | Accuracy |
|-----------------|----------|
| Healthy         | 100.00%  |
| Common Rust     | 99.31%   |
| Blight         | 95.65%   |
| Gray Leaf Spot | 90.91%   |

## Installation 

```bash
# Clone the repository
git clone https://github.com/Arian-Abdi/DeepMaize-Disease-Detection.git
cd DeepMaize-Disease-Detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## Usage 

### Training

```python
from deepmaize.model import SwinMaizeClassifier
from deepmaize.train import train_model
from deepmaize.dataset import MaizeLeafDataset, get_transforms
from torch.utils.data import DataLoader

# Initialize model
model = SwinMaizeClassifier()

# Create datasets
train_dataset = MaizeLeafDataset(
    csv_file='train.csv',
    img_dir='data/train',
    transform=get_transforms(train=True)
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Train model
train_model(model, train_loader, val_loader, num_epochs=20)
```

### Prediction

```python
from deepmaize.predict import predict_image

# Load image and predict
result = predict_image(model, 'path/to/image.jpg')
print(f"Predicted Class: {result['class']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Model Architecture 

- Base Model: Swin Transformer (swin_base_patch4_window7_224)
- Input Resolution: 224x224 pixels
- Feature Dimension: 1024
- Custom Classification Head:
  - Linear(1024 → 512)
  - ReLU
  - Dropout(0.3)
  - Linear(512 → 4)

## Contributing 

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 

- The Swin Transformer team for their groundbreaking architecture
- PyTorch team for their excellent deep learning framework
- timm library for providing pre-trained models

## Contact 

Arian Abdi - arian.abdipour9@gmail.com

Project Link: [https://github.com/Arian-Abdi/DeepMaize-Disease-Detection](https://github.com/Arian-Abdi/DeepMaize-Disease-Detection)
