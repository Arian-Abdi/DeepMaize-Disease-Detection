import torch
import torch.nn as nn
import timm

class SwinMaizeClassifier(nn.Module):
    """
    Maize leaf disease classifier using Swin Transformer architecture.
    """
    def __init__(self, num_classes=4, model_name='swin_base_patch4_window7_224'):
        super().__init__()
        
        # Load base model
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        
        # Get number of features
        n_features = self.model.num_features
        
        # Create custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Get features from Swin
        x = self.model.forward_features(x)  # [B, H, W, C]
        
        # Global average pooling over spatial dimensions
        x = x.mean(dim=[1, 2])  # Take mean over H and W dimensions
        
        # Classification
        x = self.classifier(x)
        
        return x