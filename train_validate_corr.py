import torch
from sklearn.metrics import r2_score
import numpy as np

from torch.amp import autocast


def train(model, device, train_loader, optimizer):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        device (torch.device): The device to train on.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        scaler (GradScaler): Gradient scaler for mixed precision.

    Returns:
        avg_loss (float): Average training loss.
        overall_r2 (float): R² score over the epoch.
        correlation (float): Pearson correlation coefficient over the epoch.
    """
    model.train()
    total_loss = 0.0
    all_targets = []
    all_outputs = []
    
    for batch in train_loader:
        cnn1_data, cnn2_data, y, e1, e2 = batch
        
        # Move data to device
        cnn1_data = cnn1_data.to(device, non_blocking=True)
        cnn2_data = cnn2_data.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        e1 = e1.to(device, non_blocking=True)
        e2 = e2.to(device, non_blocking=True)
        
        optimizer.zero_grad()
    
        outputs = model(cnn1_data, cnn2_data, e1, e2)
        loss = torch.nn.functional.mse_loss(outputs, y)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Collect targets and outputs for metrics
        all_targets.append(y.cpu().numpy())
        all_outputs.append(outputs.cpu().detach().numpy())
    
    avg_loss = total_loss / len(train_loader)
    all_targets = np.concatenate(all_targets).reshape(-1, 1)
    all_outputs = np.concatenate(all_outputs).reshape(-1, 1)
    overall_r2 = r2_score(all_targets, all_outputs)
    
    # Compute Pearson correlation coefficient
    if np.std(all_targets) == 0 or np.std(all_outputs) == 0:
        correlation = 0.0  # Avoid division by zero
    else:
        correlation_matrix = np.corrcoef(all_targets.flatten(), all_outputs.flatten())
        correlation = correlation_matrix[0, 1]
    
    return avg_loss, overall_r2, correlation


def evaluate(model, device, val_loader):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        device (torch.device): The device to evaluate on.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.

    Returns:
        avg_loss (float): Average validation loss.
        overall_r2 (float): R² score over the validation set.
        correlation (float): Pearson correlation coefficient over the validation set.
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in val_loader:
            cnn1_data, cnn2_data, y, e1, e2 = batch
            
            # Move data to device
            cnn1_data = cnn1_data.to(device, non_blocking=True)
            cnn2_data = cnn2_data.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            e1 = e1.to(device, non_blocking=True)
            e2 = e2.to(device, non_blocking=True)
            
            outputs = model(cnn1_data, cnn2_data, e1, e2)
            loss = torch.nn.functional.mse_loss(outputs, y)
            total_loss += loss.item()
            
            # Collect targets and outputs for metrics
            all_targets.append(y.cpu().numpy())
            all_outputs.append(outputs.cpu().detach().numpy())
    
    avg_loss = total_loss / len(val_loader)
    all_targets = np.concatenate(all_targets).reshape(-1, 1)
    all_outputs = np.concatenate(all_outputs).reshape(-1, 1)
    overall_r2 = r2_score(all_targets, all_outputs)
    
    # Compute Pearson correlation coefficient
    if np.std(all_targets) == 0 or np.std(all_outputs) == 0:
        correlation = 0.0  # Avoid division by zero
    else:
        correlation_matrix = np.corrcoef(all_targets.flatten(), all_outputs.flatten())
        correlation = correlation_matrix[0, 1]
    
    return avg_loss, overall_r2, correlation
