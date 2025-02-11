import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from src.model import SwinMaizeClassifier
from src.dataset import MaizeLeafDataset, get_transforms
from src.train import train_model
from src.predict import predict_image
from src.utils import set_seeds

def parse_args():
    parser = argparse.ArgumentParser(description='DeepMaize: Maize Leaf Disease Classification')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'],
                      help='Running mode: train or predict')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory containing the dataset')
    parser.add_argument('--train_csv', type=str, default='train.csv',
                      help='Path to training CSV file')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                      help='Path to save/load model')
    parser.add_argument('--image_path', type=str,
                      help='Path to image for prediction')
    
    return parser.parse_args()

def setup_training(args):
    """Setup training data and model"""
    # Create transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    # Create dataset
    full_dataset = MaizeLeafDataset(
        csv_file=args.train_csv,
        img_dir=Path(args.data_dir) / 'train',
        transform=train_transform
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    set_seeds(args.seed)
    
    # Initialize model
    model = SwinMaizeClassifier()
    
    if args.mode == 'train':
        # Setup training
        train_loader, val_loader = setup_training(args)
        
        # Train model
        print(f"Starting training on {args.device}...")
        train_losses, val_f1_scores = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            device=args.device
        )
        
        print("Training completed!")
        print(f"Best model saved to {args.model_path}")
        
    elif args.mode == 'predict':
        if not args.image_path:
            raise ValueError("Please provide --image_path for prediction mode")
        
        # Load model weights
        checkpoint = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(args.device)
        
        # Make prediction
        result = predict_image(model, args.image_path, device=args.device)
        
        print("\nPrediction Results:")
        print(f"Class: {result['class']}")
        print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    main()