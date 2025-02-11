import torch
from PIL import Image
from .dataset import get_transforms

class_mapping = {
    0: "Healthy",
    1: "Common Rust",
    2: "Blight",
    3: "Gray Leaf Spot"
}

def predict_image(model, image_path, device='cuda'):
    """
    Predict disease class for a single image.
    """
    # Move model to the correct device
    model = model.to(device)
    
    # Load and preprocess image
    transform = get_transforms(train=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Move image tensor to the same device as model
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    prediction = predicted.item()
    confidence_score = confidence.item()
    
    return {
        'class': class_mapping[prediction],
        'confidence': confidence_score,
        'class_id': prediction
    }

def predict_batch(model, image_paths, device='cuda'):
    """
    Predict disease classes for multiple images.
    """
    predictions = []
    for image_path in image_paths:
        pred = predict_image(model, image_path, device)
        predictions.append(pred)
    
    return predictions