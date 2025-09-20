import os
import yaml
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import mlflow

# Import your modules
from model import CPTFoundationModel, IcPredictionModel
from metrics_tracker import CPTMetricsTracker, get_gpu_memory_usage


class CPTFineTuneDataset(Dataset):
    """
    Dataset for fine-tuning that loads preprocessed features and corresponding labels.
    """
    def __init__(self, processed_dir: str, labels_df: pd.DataFrame, cpt_ids: list, 
                 max_len: int = 512, overlap: int = 128):
        """
        Args:
            processed_dir: Directory containing preprocessed .pt files
            labels_df: DataFrame with columns ['ID', 'Depth', 'Ic']
            cpt_ids: List of CPT IDs to include in this dataset
            max_len: Maximum sequence length for chunking
            overlap: Overlap between chunks
        """
        self.chunks = []
        self.labels = []
        
        print(f"Loading {len(cpt_ids)} CPT profiles for fine-tuning...")
        
        for cpt_id in tqdm(cpt_ids, desc="Loading data"):
            # Load the preprocessed features
            feature_path = os.path.join(processed_dir, f"cpt_{cpt_id}.pt")
            if not os.path.exists(feature_path):
                continue
                
            features = torch.load(feature_path)
            
            # Get the corresponding labels
            cpt_labels = labels_df[labels_df['ID'] == cpt_id].sort_values('Depth')
            
            if len(cpt_labels) == 0:
                print(f"Warning: No labels found for CPT {cpt_id}")
                continue
            
            # Convert labels to tensor
            ic_values = torch.tensor(cpt_labels['Ic'].values, dtype=torch.float32)
            
            # Ensure features and labels have same length
            min_len = min(len(features), len(ic_values))
            features = features[:min_len]
            ic_values = ic_values[:min_len]
            
            # Chunk if necessary
            if len(features) <= max_len:
                self.chunks.append(features)
                self.labels.append(ic_values)
            else:
                # Create overlapping chunks
                start = 0
                while start < len(features):
                    end = min(start + max_len, len(features))
                    self.chunks.append(features[start:end])
                    self.labels.append(ic_values[start:end])
                    
                    if end >= len(features):
                        break
                    start += max_len - overlap
        
        print(f"Created {len(self.chunks)} chunks for training")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx], self.labels[idx]


def collate_finetune(batch):
    """
    Custom collate function for fine-tuning that handles both features and labels.
    """
    features, labels = zip(*batch)
    
    # Pad features
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    
    # Pad labels with -100 (ignored by loss function)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    # Create attention mask
    lengths = [len(seq) for seq in features]
    attention_mask = torch.zeros(padded_features.shape[:2], dtype=torch.float32)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1.0
    
    return padded_features, padded_labels, attention_mask


def finetune(config: dict):
    """
    Fine-tune the foundation model for Ic prediction.
    
    Args:
        config: Configuration dictionary from YAML file
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize metrics tracker
    metrics_tracker = CPTMetricsTracker(experiment_name="cpt-finetuning", run_name="ic_prediction")
    
    # Start MLflow run and log configuration
    metrics_tracker.start_run(config)
    
    # Log system and device information using tracker
    if torch.cuda.is_available():
        gpu_memory_mb = get_gpu_memory_usage()
        gpu_memory_gb = gpu_memory_mb / 1024
        metrics_tracker.log_finetune_metrics(
            epoch=0,
            train_loss=0,
            val_loss=0,
            val_mae=0,
            val_rmse=0,
            learning_rate=0,
            epoch_time=0,
            gpu_memory_gb=gpu_memory_gb
        )
        print(f"GPU Memory Used: {gpu_memory_gb:.2f} GB")
    metrics_tracker.log_param("device", str(device))
    metrics_tracker.log_param("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        metrics_tracker.log_param("gpu_name", torch.cuda.get_device_name())
        metrics_tracker.log_param("gpu_count", torch.cuda.device_count())
    
    # Extract configuration
    paths = config['data_paths']
    foundation_params = config['foundation_model_params']
    finetune_params = config['finetuning_params']
    
    # Load the foundation model configuration
    foundation_config_path = 'configs/PG_dataset.yaml'
    with open(foundation_config_path, 'r') as f:
        foundation_config = yaml.safe_load(f)
    
    foundation_model_params = foundation_config['model_params']
    
    # --- 1. Load the pre-trained foundation model ---
    print("\nLoading pre-trained foundation model...")
    foundation_model = CPTFoundationModel(
        num_features=foundation_model_params['num_features'],
        model_dim=foundation_model_params['model_dim'],
        num_heads=foundation_model_params['num_heads'],
        num_layers=foundation_model_params['num_layers']
    )
    
    # Load pre-trained weights
    checkpoint_path = paths['foundation_model_path']
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        foundation_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded foundation model from {checkpoint_path}")
        print(f"Pre-trained for {checkpoint.get('epoch', 'unknown')} epochs")
    else:
        raise FileNotFoundError(f"Foundation model not found at {checkpoint_path}")
    
    # --- 2. Create the fine-tuning model ---
    print("\nCreating fine-tuning model...")
    model = IcPredictionModel(
        foundation_model=foundation_model,
        model_dim=foundation_model_params['model_dim']
    ).to(device)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (regression head): {trainable_params:,}")
    print(f"Frozen parameters (foundation): {total_params - trainable_params:,}")
    
    # Log model parameters to MLflow using tracker
    metrics_tracker.log_param("total_params", total_params)
    metrics_tracker.log_param("trainable_params", trainable_params)
    metrics_tracker.log_param("frozen_params", total_params - trainable_params)
    
    # --- 3. Load labels ---
    print("\nLoading labels...")
    labels_df = pd.read_csv(paths['labels_file'])
    print(f"Loaded {len(labels_df)} label entries")
    print(f"Ic statistics: mean={labels_df['Ic'].mean():.2f}, std={labels_df['Ic'].std():.2f}")
    
    # --- 4. Create datasets ---
    print("\nCreating datasets...")
    
    # Load split IDs
    splits_dir = 'data/splits'
    train_ids = np.loadtxt(os.path.join(splits_dir, 'train_ids.txt'), dtype=int).tolist()
    val_ids = np.loadtxt(os.path.join(splits_dir, 'val_ids.txt'), dtype=int).tolist()
    test_ids = np.loadtxt(os.path.join(splits_dir, 'test_ids.txt'), dtype=int).tolist()
    
    # Create datasets
    train_dataset = CPTFineTuneDataset(
        paths['processed_dir'], labels_df, train_ids, 
        max_len=512, overlap=128
    )
    val_dataset = CPTFineTuneDataset(
        paths['processed_dir'], labels_df, val_ids,
        max_len=512, overlap=128
    )
    
    # Create dataloaders
    batch_size = finetune_params['batch_size']
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_finetune, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_finetune, num_workers=4, pin_memory=True
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Log dataset information using tracker
    metrics_tracker.log_param("train_size", len(train_dataset))
    metrics_tracker.log_param("val_size", len(val_dataset))
    metrics_tracker.log_param("train_batches", len(train_loader))
    metrics_tracker.log_param("val_batches", len(val_loader))
    
    # --- 5. Setup training ---
    learning_rate = finetune_params['learning_rate']
    num_epochs = finetune_params['num_epochs']
    
    # Only optimize the regression head parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Use MSE loss for regression, with masking for padded values
    def masked_mse_loss(predictions, targets, mask):
        """Calculate MSE loss only for non-padded positions."""
        # Mask out padded positions (where target is -100)
        valid_mask = (targets != -100) & (mask == 1)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        valid_predictions = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        
        return nn.functional.mse_loss(valid_predictions, valid_targets)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3,
    )
    
    # --- 6. Training loop ---
    print(f"\nStarting fine-tuning for {num_epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    
    best_val_loss = float('inf')
    save_path = paths['finetuned_model_save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for features, labels, attention_mask in pbar:
                features = features.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)
                
                # Forward pass
                predictions = model(features, attention_mask)
                
                # Calculate loss
                loss = masked_mse_loss(predictions, labels, attention_mask)
                
                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss detected, skipping batch")
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 
                    max_norm=1.0
                )
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                train_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, labels, attention_mask in tqdm(val_loader, desc="Validation", leave=False):
                features = features.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)
                
                predictions = model(features, attention_mask)
                loss = masked_mse_loss(predictions, labels, attention_mask)
                
                if torch.isfinite(loss):
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # Store predictions for additional metrics
                    valid_mask = (labels != -100) & (attention_mask == 1)
                    if valid_mask.any():
                        all_predictions.extend(predictions[valid_mask].cpu().numpy())
                        all_targets.extend(labels[valid_mask].cpu().numpy())
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        
        # Calculate additional metrics
        if all_predictions:
            predictions_np = np.array(all_predictions)
            targets_np = np.array(all_targets)
            mae = np.mean(np.abs(predictions_np - targets_np))
            rmse = np.sqrt(np.mean((predictions_np - targets_np) ** 2))
        else:
            mae = rmse = float('inf')
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics to MLflow using the tracker
        gpu_memory_gb = None
        if torch.cuda.is_available():
            gpu_memory_mb = get_gpu_memory_usage()
            gpu_memory_gb = gpu_memory_mb / 1024
        
        metrics_tracker.log_finetune_metrics(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_mae=mae,
            val_rmse=rmse,
            learning_rate=current_lr,
            epoch_time=epoch_time,
            gpu_memory_gb=gpu_memory_gb
        )
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss (MSE): {avg_train_loss:.6f}")
        print(f"  Val Loss (MSE): {avg_val_loss:.6f}")
        print(f"  Val MAE: {mae:.4f}")
        print(f"  Val RMSE: {rmse:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        print(f"  Epoch Time: {epoch_time:.1f}s")
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_mae': mae,
                'val_rmse': rmse,
                'config': config
            }, save_path)
            print(f"  ✓ Saved best model with validation loss: {best_val_loss:.6f}")
    
    # Log final summary, plots, and metrics using tracker
    metrics_tracker.log_final_summary(best_val_loss, num_epochs)
    metrics_tracker.create_finetune_plots()
    metrics_tracker.export_metrics_csv("finetune_metrics.csv")
    # Log the trained model
    try:
        metrics_tracker.log_model_checkpoint(model, save_path, is_best=True)
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")
    metrics_tracker.end_run()
    
    print(f"\n{'='*50}")
    print(f"Fine-tuning complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {save_path}")
    print(f"MLflow run completed. View results with: mlflow ui")
    print(f"{'='*50}")


if __name__ == '__main__':
    DEFAULT_CONFIG_PATH = 'configs/PG_finetune_Ic.yaml'
    parser = argparse.ArgumentParser(description="Fine-tune CPT foundation model for Ic prediction")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help="Path to fine-tuning configuration file")
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {args.config}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)
    
    finetune(config)