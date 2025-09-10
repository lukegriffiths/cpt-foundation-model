import os
import yaml
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

# Import the new DataModule
from data_utils import CPTDataModule 
from model import CPTFoundationModel

from torch.cuda.amp import autocast, GradScaler


def train(config: dict):
    """
    Main function to run the model training pipeline.
    
    Args:
        config (dict): A dictionary containing all configuration parameters.
    """
    # --- 1. Setup and Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Extract paths and create directories
    paths = config['data_paths']
    model_save_path = paths['model_save_path']
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Extract training and model hyperparameters
    train_params = config['training_params']
    model_params = config['model_params']
    
    # Get Training Parameters
    learning_rate = train_params['learning_rate']
    num_epochs = train_params['num_epochs']
    mask_ratio = train_params.get('mask_ratio', 0.15)
    
    print(f"Training parameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Mask ratio: {mask_ratio}")
    print(f"  Batch size: {train_params['batch_size']}")
    print(f"  Model dim: {model_params['model_dim']}")
    print(f"  Num heads: {model_params['num_heads']}")
    print(f"  Num layers: {model_params['num_layers']}")

    # --- 2. Data Loading ---
    print("\nSetting up data module...")
    data_module = CPTDataModule(config)
    data_module.setup()
    
    # Get DataLoaders for train and validation
    train_loader = data_module.get_dataloader(stage='train', shuffle=True)
    val_loader = data_module.get_dataloader(stage='val', shuffle=False)
    
    print("Data loading complete.")

    # --- 3. Model Initialization ---
    model = CPTFoundationModel(
        num_features=model_params['num_features'],
        model_dim=model_params['model_dim'],
        num_heads=model_params['num_heads'],
        num_layers=model_params['num_layers'],
        dropout=model_params.get('dropout', 0.1)
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Use L1 loss which is more robust to outliers than MSE
    loss_fn = nn.L1Loss()
    
    # Initialize grad scaler for mixed precision training
    scaler = GradScaler()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    print("\nStarting training...")
    best_loss = float('inf')
    
    # --- 4. Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, (batch, attention_mask) in enumerate(pbar):
                batch = batch.to(device)
                attention_mask = attention_mask.to(device)
                
                # Create masked input
                corrupted_batch = batch.clone()
                
                # Random mask for positions to predict
                prob_mask = torch.rand(batch.shape[:2], device=device)
                masking_condition = (prob_mask < mask_ratio) & (attention_mask == 1)
                
                num_masked = masking_condition.sum().item()
                if num_masked == 0:
                    continue
                
                # Use small random noise for masked positions (helps learning)
                mask_noise = torch.randn_like(batch) * 0.1
                corrupted_batch = torch.where(
                    masking_condition.unsqueeze(-1), 
                    mask_noise, 
                    corrupted_batch
                )
                
                # Forward Pass with mixed precision
                optimizer.zero_grad()
                
                with autocast(): # this enables mixed precision
                    # Get contextual embeddings from the model
                    contextual_embeddings = model(corrupted_batch, attention_mask)
                    
                    # Apply final projection ONLY to masked tokens
                    masked_embeddings = contextual_embeddings[masking_condition]
                    predictions = model.output_projection(masked_embeddings)
                    
                    # Get target values and calculate loss
                    target_values = batch[masking_condition]
                    loss = loss_fn(predictions, target_values)
                
                # Check for NaN
                if not torch.isfinite(loss):
                    print(f"\nWarning: Non-finite loss detected at batch {batch_idx}, skipping...")
                    continue
                
                # Backward Pass
                scaler.scale(loss).backward()
                
                # Gradient clipping, this helps stabilize training
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}' if num_batches > 0 else 'N/A'
                })
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"\nEpoch {epoch+1}/{num_epochs} - Training Loss: {avg_loss:.6f}")
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch, attention_mask in tqdm(val_loader, desc="Validation", leave=False):
                    batch = batch.to(device)
                    attention_mask = attention_mask.to(device)
                    
                    # Same masking process as training
                    corrupted_batch = batch.clone()
                    prob_mask = torch.rand(batch.shape[:2], device=device)
                    masking_condition = (prob_mask < 0.15) & (attention_mask == 1)
                    
                    if masking_condition.sum() == 0:
                        continue
                    
                    mask_noise = torch.randn_like(batch) * 0.1
                    corrupted_batch = torch.where(
                        masking_condition.unsqueeze(-1), 
                        mask_noise, 
                        corrupted_batch
                    )
                    
                    # Forward pass
                    contextual_embeddings = model(corrupted_batch, attention_mask)
                    masked_embeddings = contextual_embeddings[masking_condition]
                    predictions = model.output_projection(masked_embeddings)
                    
                    loss = loss_fn(predictions, batch[masking_condition])
                    
                    if torch.isfinite(loss):
                        val_loss += loss.item()
                        val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            print(f"Validation Loss: {avg_val_loss:.6f}")
            
            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                    'config': config
                }, model_save_path.replace('.pth', '_best.pth'))
                print(f"Saved best model with validation loss: {best_loss:.6f}")
        
        # Regular checkpoint save
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, model_save_path)

    print(f"\nTraining complete. Models saved to '{model_save_path}'")


if __name__ == '__main__':
    DEFAULT_CONFIG_PATH = 'configs/PG_dataset.yaml'
    parser = argparse.ArgumentParser(description="Train a CPT Foundation Model.")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        exit()
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit()
    
    train(config)