import os
import yaml
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

# Import your modules
from data_utils import CPTDataModule 
from model import CPTFoundationModel
from masking import create_span_mask, create_block_mask, apply_mask_to_input

from torch.cuda.amp import autocast, GradScaler


def train(config: dict):
    """
    Main function to run the model training pipeline with improved masking.
    
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
    
    # Masking parameters
    mask_ratio = train_params.get('mask_ratio', 0.15)
    mask_strategy = train_params.get('mask_strategy', 'span')  # 'random', 'span', or 'block'
    mask_type = train_params.get('mask_type', 'noise')  # 'noise', 'zero', 'mean'
    mean_span_length = train_params.get('mean_span_length', 5)
    block_size = train_params.get('block_size', 10)
    
    print(f"\nTraining Configuration:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {train_params['batch_size']}")
    print(f"  Model dim: {model_params['model_dim']}")
    print(f"  Num heads: {model_params['num_heads']}")
    print(f"  Num layers: {model_params['num_layers']}")
    print(f"\nMasking Configuration:")
    print(f"  Strategy: {mask_strategy}")
    print(f"  Mask ratio: {mask_ratio}")
    print(f"  Mask type: {mask_type}")
    if mask_strategy == 'span':
        print(f"  Mean span length: {mean_span_length}")
    elif mask_strategy == 'block':
        print(f"  Block size: {block_size}")

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
        total_masked = 0
        total_tokens = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, (batch, attention_mask) in enumerate(pbar):
                batch = batch.to(device)
                attention_mask = attention_mask.to(device)
                
                # --- Create mask based on selected strategy ---
                if mask_strategy == 'random':
                    # Original random masking
                    prob_mask = torch.rand(batch.shape[:2], device=device)
                    masking_condition = (prob_mask < mask_ratio) & (attention_mask == 1)
                
                elif mask_strategy == 'span':
                    # Span-based masking (recommended for CPT data)
                    masking_condition = create_span_mask(
                        batch.shape[:2], 
                        attention_mask, 
                        mask_ratio, 
                        mean_span_length, 
                        device
                    )
                
                elif mask_strategy == 'block':
                    # Fixed block masking
                    masking_condition = create_block_mask(
                        batch.shape[:2], 
                        attention_mask, 
                        mask_ratio, 
                        block_size, 
                        device
                    )
                
                else:
                    raise ValueError(f"Unknown mask strategy: {mask_strategy}")
                
                # Check if any tokens are masked
                num_masked = masking_condition.sum().item()
                if num_masked == 0:
                    continue
                
                # Track masking statistics
                num_valid = attention_mask.sum().item()
                total_masked += num_masked
                total_tokens += num_valid
                
                # --- Apply masking to input ---
                corrupted_batch = apply_mask_to_input(
                    batch, 
                    masking_condition, 
                    mask_type=mask_type,
                    mask_value=None  # Not using learnable mask for now
                )
                
                # --- Forward Pass with mixed precision ---
                optimizer.zero_grad()
                
                with autocast():
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
                
                # --- Backward Pass ---
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                actual_mask_ratio = total_masked / total_tokens if total_tokens > 0 else 0
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}' if num_batches > 0 else 'N/A',
                    'mask_ratio': f'{actual_mask_ratio:.3f}'
                })
                
                # Debug: Print sample predictions every 100 batches
                if batch_idx % 100 == 0 and batch_idx > 0 and epoch == 0:
                    with torch.no_grad():
                        sample_pred = predictions[:3].cpu()
                        sample_target = target_values[:3].cpu()
                        print(f"\n  Sample prediction: {sample_pred[0].numpy()}")
                        print(f"  Sample target:     {sample_target[0].numpy()}")
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        actual_mask_ratio = total_masked / total_tokens if total_tokens > 0 else 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Training Loss: {avg_loss:.6f}")
        print(f"  Actual mask ratio: {actual_mask_ratio:.3f}")
        
        # --- Validation every 5 epochs ---
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            val_masked = 0
            val_tokens = 0
            
            with torch.no_grad():
                for batch, attention_mask in tqdm(val_loader, desc="Validation", leave=False):
                    batch = batch.to(device)
                    attention_mask = attention_mask.to(device)
                    
                    # Use same masking strategy as training
                    if mask_strategy == 'random':
                        prob_mask = torch.rand(batch.shape[:2], device=device)
                        masking_condition = (prob_mask < mask_ratio) & (attention_mask == 1)
                    elif mask_strategy == 'span':
                        masking_condition = create_span_mask(
                            batch.shape[:2], attention_mask, 
                            mask_ratio, mean_span_length, device
                        )
                    elif mask_strategy == 'block':
                        masking_condition = create_block_mask(
                            batch.shape[:2], attention_mask,
                            mask_ratio, block_size, device
                        )
                    
                    if masking_condition.sum() == 0:
                        continue
                    
                    val_masked += masking_condition.sum().item()
                    val_tokens += attention_mask.sum().item()
                    
                    # Apply same masking type
                    corrupted_batch = apply_mask_to_input(
                        batch, 
                        masking_condition, 
                        mask_type=mask_type,
                        attention_mask=attention_mask
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
            val_mask_ratio = val_masked / val_tokens if val_tokens > 0 else 0
            
            print(f"  Validation Loss: {avg_val_loss:.6f}")
            print(f"  Validation mask ratio: {val_mask_ratio:.3f}")
            
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
                print(f"  ✓ Saved best model with validation loss: {best_loss:.6f}")
        
        # Regular checkpoint save every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, model_save_path)
            print(f"  ✓ Saved checkpoint at epoch {epoch+1}")

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Models saved to: {os.path.dirname(model_save_path)}")
    print(f"{'='*50}")


if __name__ == '__main__':
    DEFAULT_CONFIG_PATH = 'configs/PG_dataset.yaml'
    parser = argparse.ArgumentParser(description="Train a CPT Foundation Model with advanced masking.")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {args.config}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit(1)
    
    train(config)