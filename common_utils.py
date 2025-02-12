from torch.utils.data import Subset
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
import csv
from datetime import datetime


def initialize_logger(directory):
    """Initialize the CSV logger."""
    os.makedirs(directory, exist_ok=True)
    log_file_path = os.path.join(directory, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_training_log.csv")
    with open(log_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fold', 'Epoch', 'Train Loss', 'Train R2', 'Train_Corr', 'Validation Loss', 'Validation R2', 'Validation_Corr'])
    return log_file_path

def log_metrics(log_file_path, metrics):
    """Log metrics to the CSV file."""
    with open(log_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(metrics)

def split_dataset(dataset):
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset

def load_checkpoint(fold, directory):
    """Load a checkpoint file from a specified directory."""
    fold_directory = os.path.join(directory, f"fold_{fold}")
    filepath = os.path.join(fold_directory, "model_checkpoint.pth")    
    if os.path.isfile(filepath):
        return torch.load(filepath, weights_only=True )
    else:
        print("Existing checpoint not found")
    return None

def save_checkpoint(state, directory, filename = "model_checkpoint.pth" ):
    """Save the current state of the model, optimizer, and other parameters."""
    
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename) 
    torch.save(state, filepath)
    
def init_weights(m):
    """
    Initialize weights for Conv3d and Linear layers using Kaiming Normal initialization.
    """
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)