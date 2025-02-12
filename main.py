import torch
import torch.nn as nn
import shutil
from torch.optim import Adam
from sklearn.model_selection import KFold
from networks.dual_cnn_regressor import DualCNNRegressor
from torch.utils.data import DataLoader, Subset
from dataset_factory.sMRI_Dataset_Dense import MRIDataset
from train_validate_corr import train, evaluate
from tqdm import tqdm
from common_utils import split_dataset, initialize_logger, log_metrics, load_checkpoint, save_checkpoint, init_weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import argparse
import torch.multiprocessing as mp
from datetime import datetime
import yaml
import wandb

def load_config(config_path: str) -> dict:
    """
    Load parameters from a YAML configuration file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config




# ----------------------------
# Main Training Script
# ----------------------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train Dual CNN Regressor with Attention")
    
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file."
    )
    
    args = parser.parse_args()
    # 2.1 Load Config
    config = load_config(args.config_path)
    print("-------------- Loaded Config --------------")
    print(config)

    # ----------------------------------------
    # 2.2 Parse Model and Training Params
    # ----------------------------------------
    model_name = config["model"]["model_name"]
    dropout_rate = float(config["model"]["dropout_rate"])
    num_classes = int(config["model"]["num_classes"])
    use_pos_encoding = config["model"]["use_pos_encoding"]
    use_attention = config["model"]["use_attention"]
    normalization = config["model"]["normalization"]

    base_lr = float(config["training"]["lr"])
    early_stop_patience = config["training"]["early_stop_patience"]
    batch_size = config["training"]["batch_size"]
    loss_type = config["training"]["loss_type"]
    num_epochs = config["training"]["epochs"]
    lr_decay_patience = config["training"]["lr_decay_patience"]

    # Additional optional warmup
    warmup_epochs = config["training"].get("warmup_steps") 

    # ----------------------------------------
    # 2.3 Prepare Experiment Directory
    # ----------------------------------------
    experiments_dir = "experiments"
    model_directory = os.path.join(experiments_dir, model_name)

    if os.path.exists(model_directory):
        print(f"[Info] Using existing model directory: {model_directory}")
    else:
        os.makedirs(model_directory, exist_ok=True)
        print(f"[Info] Created new model directory: {model_directory}")
        
    #  2.4 Save the Config File Inside the Model Folder
    try:
        config_save_path = os.path.join(model_directory, "config.yaml")
        shutil.copy(args.config_path, config_save_path)
        print(f"[Info] Saved config.yaml to: {config_save_path}")
    except Exception as e:
        print(f"[Warning] Could not save config.yaml: {e}")

    # Create fold subdirectories and logs subdirectory
    for i in range(5):
        fold_directory = os.path.join(model_directory, f"fold_{i}")
        os.makedirs(fold_directory, exist_ok=True)

    training_log_directory = os.path.join(model_directory, "training_logs")
    os.makedirs(training_log_directory, exist_ok=True)
    print(f"[Info] Subdirectories prepared in: {model_directory}")
    # Initialize Logger
    log_file_path = initialize_logger(training_log_directory)

    # ----------------------------------------
    # 2.4 Device, Dataset, and Splits
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Running on device: {device}")

    # Adjust the dataset path as necessary
    dataset_path = "/data/users4/sdeshpande8/precomputed_tensors_working_memory/" 
    dataset = MRIDataset(precomputed_dir=dataset_path)
    train_dataset, test_dataset = split_dataset(dataset)

    # K-Fold setup
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    
    
        # ----------------------------------------
    # 2.5 Initialize WandB
    # ----------------------------------------
    wandb.login(key="ac997d805f2d6befc3c1f8f32d191fd93cb04b03")
    wandb.init(project="Dense_Region_Aware_VIT", name=model_name)
    
    
    # ====================================
    # 2.5 Training Over 5 Folds
    # ====================================
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print("===================================")
        print(f"         Starting Fold {fold}      ")
        print("===================================")
        
        # If this fold is already marked complete, skip
        completed_path = os.path.join(model_directory, f"fold_{fold}", "Completed.pth")
        if os.path.isfile(completed_path):
            print(f"[Info] Fold {fold} is already complete. Skipping.")
            continue
        
        
        
        dual_regressor = DualCNNRegressor(dropout_rate = dropout_rate,
                                          num_classes = num_classes,
                                          use_pos_encoding = use_pos_encoding,
                                          use_attention = use_attention,
                                          normalization = normalization ).to(device)
        
        dual_regressor = nn.DataParallel(dual_regressor)

        # Apply custom weight initialization
        dual_regressor.apply(init_weights)

        optimizer = Adam(dual_regressor.parameters(), lr=base_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_decay_patience)
        start_epoch = 0
        best_val_loss = float('inf')
        best_model_state = None

        # ----------------------------
        # Early Stopping Parameters
        # ----------------------------
        early_stop_patience = 10  # number of epochs to wait
        patience_counter = 0

        # Check if fold training is already complete
        if os.path.isfile(f"{model_directory}/fold_{fold}/Completed.pth"):
            print(f'Fold {fold} training is complete, moving over')
            continue

        checkpoint = load_checkpoint(fold, model_directory)
        if checkpoint:
            dual_regressor.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_model_state = dual_regressor.state_dict()
            best_val_loss = checkpoint['val_loss']
            print(f"Resuming training from epoch {start_epoch}")

        train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(Subset(train_dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Epochs", position=0):
            # ----------------------------
            # Learning Rate Warmup
            # ----------------------------
            if epoch < warmup_epochs:
                # Linearly increase LR from a small value to the base_lr
                current_lr = base_lr * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                print(f"Warmup Epoch {epoch}: Setting LR to {current_lr:.6f}")
            else:
                # For ReduceLROnPlateau, we call scheduler.step(val_loss) after validation.
                pass

            # ----------------------------
            # Training and Validation
            # ----------------------------
            train_loss, train_r2, train_corr = train(dual_regressor, device, train_loader, optimizer)
            val_loss, val_r2, val_corr = evaluate(dual_regressor, device, val_loader)
            log_metrics(log_file_path, [fold, epoch, train_loss, train_r2, train_corr, val_loss, val_r2, val_corr])
            print(f"Train_Loss: {train_loss} and Train R2: {train_r2}")
            print(f"Val_Loss: {val_loss} and Val R2: {val_r2}")
            
            
            
            wandb.log({
                f"Fold_{fold}/train_loss": train_loss,
                f"Fold_{fold}/train_r2": train_r2,
                f"Fold_{fold}/val_loss": val_loss,
                f"Fold_{fold}/val_r2": val_r2,
                "epoch": epoch,
                "learning_rate": current_lr
            })

            # ----------------------------
            # Early Stopping Logic and Checkpoint Saving
            # ----------------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving a new best model for fold {fold} at epoch {epoch}")
                best_model_state = dual_regressor.state_dict()
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, directory=f"{model_directory}/fold_{fold}")
                # Reset early stopping counter when improvement occurs
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Early stopping counter: {patience_counter} / {early_stop_patience}")
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping triggered for fold {fold} at epoch {epoch}")
                    save_checkpoint({
                    'epoch': epoch,
                    }, directory=f"{model_directory}/fold_{fold}", filename='Completed.pth')
                    break

            if epoch >= warmup_epochs:
                scheduler.step(val_loss)

            if epoch == num_epochs:
                print(f'Training for fold {fold} is complete')
                save_checkpoint({
                    'epoch': epoch,
                }, directory=f"{model_directory}/fold_{fold}", filename='Completed.pth')
        print('----------------------------------------------------------------------------------------')

    print("Training complete. Evaluating on test data...")
