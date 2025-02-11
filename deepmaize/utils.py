import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_confusion_matrix(model, val_loader, device='cuda'):
    """Generate and plot confusion matrix."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["Healthy", "Common Rust", "Blight", "Gray Leaf Spot"],
        yticklabels=["Healthy", "Common Rust", "Blight", "Gray Leaf Spot"]
    )
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    return plt.gcf()

def visualize_predictions(model, images, device='cuda'):
    """Visualize model predictions on a batch of images."""
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    # Create grid of images
    fig, axes = plt.subplots(1, min(4, len(images)), figsize=(15, 4))
    if len(images) == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        img = images[i].permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f"Pred: {class_mapping[predicted[i].item()]}\nConf: {confidence[i].item():.2f}")
        ax.axis('off')
    
    plt.tight_layout()
    return fig