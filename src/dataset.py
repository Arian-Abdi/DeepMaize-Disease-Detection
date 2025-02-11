import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms

class MaizeLeafDataset(Dataset):
    """
    Dataset class for maize leaf disease classification.
    """
    def __init__(self, csv_file, img_dir, transform=None):
        self.leaf_data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.leaf_data)
    
    def __getitem__(self, idx):
        img_name = f"{self.leaf_data.iloc[idx]['Image']}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load and convert image to RGB
        image = Image.open(img_path).convert('RGB')
        label = self.leaf_data.iloc[idx]['Label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(train=True):
    """
    Get data transforms for training or validation.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])