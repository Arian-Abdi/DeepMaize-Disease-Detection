import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    """
    Train the model with the given parameters.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Training metrics
    best_val_f1 = 0.0
    train_losses = []
    val_accuracies = []
    val_f1_scores = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_predictions = []
        train_labels = []
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # Collect predictions for F1 score
            _, predicted = torch.max(outputs.data, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate training F1 score
        train_f1 = f1_score(
            np.array(train_labels), 
            np.array(train_predictions), 
            average='weighted'
        ) * 100
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(
            np.array(val_labels), 
            np.array(val_predictions), 
            average='weighted'
        ) * 100
        
        train_losses.append(epoch_loss)
        val_f1_scores.append(val_f1)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]:')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Training F1 Score: {train_f1:.2f}%')
        print(f'Validation Loss: {epoch_val_loss:.4f}')
        print(f'Validation F1 Score: {val_f1:.2f}%')
        
        # Save best model based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1_score': val_f1,
            }, 'best_model.pth')
            print(f'\nBest model saved with F1 Score: {best_val_f1:.2f}%')
        
        scheduler.step()
    
    return train_losses, val_f1_scores