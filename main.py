import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import KFold
from networks.dual_cnn_regressor import DualCNNRegressor
from torch.utils.data import DataLoader, Subset
from MRI_Dataset_Simple import MRIDataset
from train_validate_corr import train, evaluate
from tqdm import tqdm
from common_utils import split_dataset, initialize_logger, log_metrics, load_checkpoint, save_checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import argparse
from datetime import datetime


if __name__ == '__main__':
    
    #mp.set_start_method('spawn')
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
        training_log_directory = os.path.join(model_directory,"training_logs")
        os.makedirs(training_log_directory, exist_ok=True)

        print(f"Pre-created subdirectories for folds and logs in {model_directory}")
        
    
    batch_size  = 8
    num_epochs = 100
    metric_directory = f"{model_directory}/training_logs"
    log_file_path = initialize_logger(metric_directory)
    print('Main Function started')
    print('------------------------------------------------------------------------')
    device = torch.device('cuda')
    print(device)
    print("Initializing DataSet")
    dataset = MRIDataset(precomputed_dir="/data/users4/sdeshpande8/precomputed_tensors_working_memory/")
    print(" DataSet Initialized")
    train_dataset, test_dataset = split_dataset(dataset)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'Starting Fold {fold}')
        dual_regressor = DualCNNRegressor().to(device)
        dual_regressor = nn.DataParallel(dual_regressor)
            
        optimizer = Adam(dual_regressor.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)
        start_epoch = 0
        best_val_loss = float('inf')
        best_model_state = None
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
        
        for epoch in tqdm(range(start_epoch, num_epochs+1), desc="Epochs", position=0):
            train_loss, train_r2, train_corr = train(dual_regressor, device, train_loader, optimizer)
            val_loss, val_r2, val_corr = evaluate(dual_regressor, device, val_loader)
            log_metrics(log_file_path, [fold, epoch, train_loss, train_r2, train_corr, val_loss, val_r2, val_corr])
            print(f"Train_Loss:{train_loss} and Train R2:{train_r2}")
            print(f"Val_Loss:{val_loss} and Val R2:{val_r2}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving a new best model for fold{fold} in epoch {epoch}")
                best_model_state = dual_regressor.state_dict()
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, directory=f"{model_directory}/fold_{fold}")

            scheduler.step(val_loss)
                
            if epoch==num_epochs:
                print(f'Training for fold{fold} is complete')     
                save_checkpoint({
                'epoch': epoch,
                }, directory=f"{model_directory}/fold_{fold}",  filename = 'Completed.pth')
        print('----------------------------------------------------------------------------------------')

    print("Training complete. Evaluating on test data...")