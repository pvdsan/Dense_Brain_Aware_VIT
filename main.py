import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import KFold
from networks.dual_cnn_regressor import DualCNNRegressor
from torch.utils.data import DataLoader, Subset
from sMRI_Dataset import MRIDataset
from train_validate_corr import train, evaluate
from tqdm import tqdm
from common_utils import split_dataset, initialize_logger, log_metrics, load_checkpoint, save_checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import argparse
import torch.multiprocessing as mp
from datetime import datetime

# ----------------------------
# Weight Initialization Method
# ----------------------------
def init_weights(m):
    """
    Initialize weights for Conv3d and Linear layers using Kaiming Normal initialization.
    """
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# ----------------------------
# Main Training Script
# ----------------------------
if __name__ == '__main__':
    # mp.set_start_method('spawn')
    experiments_dir = "experiments"

    parser = argparse.ArgumentParser(description="Train Dual CNN Regressor with Attention")
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help="Specify the model directory for saving/loading checkpoints. If it exists, old checkpoints will be used; otherwise, a new directory will be created."
    )
    args = parser.parse_args()

    model_directory = os.path.join(experiments_dir, args.model_name)

    # Check and create model directory if needed
    if os.path.exists(model_directory):
        print(f"Using existing model directory: {model_directory}")
    else:
        os.makedirs(model_directory, exist_ok=True)
        print(f"Created new model directory: {model_directory}")

    # Pre-create subdirectories for folds and the best model
    for i in range(5):  # Assuming 5 folds
        fold_directory = os.path.join(model_directory, f"fold_{i}")
        os.makedirs(fold_directory, exist_ok=True)

    training_log_directory = os.path.join(model_directory, "training_logs")
    os.makedirs(training_log_directory, exist_ok=True)

    print(f"Pre-created subdirectories for folds and logs in {model_directory}")

    batch_size  = 8
    num_epochs = 100
    warmup_epochs = 5  # Number of epochs for warmup
    base_lr = 1e-3    # Base learning rate after warmup

    metric_directory = f"{model_directory}/training_logs"
    log_file_path = initialize_logger(metric_directory)
    print('Main Function started')
    print('------------------------------------------------------------------------')
    device = torch.device('cuda')
    print(device)
    print("Initializing DataSet")
    dataset = MRIDataset(precomputed_dir="/data/users4/sdeshpande8/precomputed_tensors_working_memory/")
    print("DataSet Initialized")
    train_dataset, test_dataset = split_dataset(dataset)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'Starting Fold {fold}')
        dual_regressor = DualCNNRegressor().to(device)
        dual_regressor = nn.DataParallel(dual_regressor)

        # Apply custom weight initialization
        dual_regressor.apply(init_weights)

        optimizer = Adam(dual_regressor.parameters(), lr=base_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        start_epoch = 0
        best_val_loss = float('inf')
        best_model_state = None

        # ----------------------------
        # Early Stopping Parameters
        # ----------------------------
        patience = 10  # number of epochs to wait
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
                print(f"Early stopping counter: {patience_counter} / {patience}")
                if patience_counter >= patience:
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
