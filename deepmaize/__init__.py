# Package metadata
__version__ = "1.0.0"
__author__ = "Arian Abdi"
__description__ = "Deep learning-powered maize leaf disease detection using Swin Transformer"

# Import main components
from .model import SwinMaizeClassifier
from .dataset import MaizeLeafDataset, get_transforms
from .train import train_model
from .predict import predict_image
from .utils import set_seeds, plot_confusion_matrix, visualize_predictions

# Define constants
SUPPORTED_CLASSES = {
    0: "Healthy",
    1: "Common Rust",
    2: "Blight",
    3: "Gray Leaf Spot"
}

# Default model configuration
MODEL_CONFIG = {
    'model_name': 'swin_base_patch4_window7_224',
    'num_classes': len(SUPPORTED_CLASSES),
    'feature_dim': 1024,
    'dropout': 0.3
}

# Default training configuration
TRAINING_CONFIG = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'num_epochs': 20,
    'train_val_split': 0.8
}

# Define public API
__all__ = [
    # Main classes
    'SwinMaizeClassifier',
    'MaizeLeafDataset',
    
    # Functions
    'train_model',
    'predict_image',
    'get_transforms',
    'set_seeds',
    'plot_confusion_matrix',
    'visualize_predictions',
    
    # Constants
    'SUPPORTED_CLASSES',
    'MODEL_CONFIG',
    'TRAINING_CONFIG',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]