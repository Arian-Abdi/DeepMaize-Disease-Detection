# DeepMaize-Disease-Detection
Advanced computer vision system for detecting and classifying maize leaf diseases using state-of-the-art deep learning.

# DeepMaize Disease Detection 

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep learning-powered maize leaf disease detection using the Swin Transformer architecture. This model achieves high accuracy in classifying four distinct leaf conditions: Healthy, Common Rust, Blight, and Gray Leaf Spot.

##  Key Features

- **High Accuracy**: 97%+ accuracy across all disease classes
- **State-of-the-Art Architecture**: Utilizes Swin Transformer for superior feature extraction
- **Production Ready**: Complete with data preprocessing, training, and inference pipelines
- **Easy to Use**: Simple API for both training and prediction

##  Performance

| Disease Class    | Accuracy |
|-----------------|----------|
| Healthy         | 100.00%  |
| Common Rust     | 99.89%   |
| Blight         | 96.46%   |
| Gray Leaf Spot | 93.12%   |

##  Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/Arian-Abdi/DeepMaize-Disease-Detection.git
cd DeepMaize-Disease-Detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python main.py --mode train \
               --data_dir data \
               --train_csv train.csv \
               --num_epochs 20
```

### Prediction

```bash
python main.py --mode predict \
               --model_path models/best_model.pth \
               --image_path path/to/image.jpg
```

##  Project Structure

```
deepmaize/
├── src/
│   ├── model.py          # Model architecture
│   ├── dataset.py        # Data loading and preprocessing
│   ├── train.py          # Training functions
│   ├── predict.py        # Inference pipeline
│   └── utils.py          # Helper functions
├── main.py              # Main entry point
└── requirements.txt     # Dependencies
```

##  Configuration

Key parameters that can be configured via command line arguments:

```bash
--batch_size      # Training batch size (default: 16)
--learning_rate   # Learning rate (default: 1e-4)
--num_epochs      # Number of training epochs (default: 20)
--device          # Device to use (default: 'cuda' if available)
```

##  Model Architecture

- Base Model: Swin Transformer (swin_base_patch4_window7_224)
- Input Resolution: 224x224 pixels
- Feature Dimension: 1024
- Custom Classification Head:
  - Linear(1024 → 512)
  - ReLU
  - Dropout(0.3)
  - Linear(512 → 4)

##  Dataset

The model is trained on a dataset of maize leaf images with the following distribution:
- Common Rust: 1,073 images (32.06%)
- Healthy: 917 images (27.40%)
- Blight: 908 images (27.13%)
- Gray Leaf Spot: 449 images (13.41%)

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

##  Contact

Arian Abdi - [Your Email]

Project Link: [https://github.com/Arian-Abdi/DeepMaize-Disease-Detection](https://github.com/Arian-Abdi/DeepMaize-Disease-Detection)

## 🙏 Acknowledgments

- The Swin Transformer team for their groundbreaking architecture
- PyTorch team for their excellent deep learning framework
- timm library for providing pre-trained models
