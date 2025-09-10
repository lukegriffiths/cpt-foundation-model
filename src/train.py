import os
import yaml
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

# Import the new DataModule
from data_utils import CPTDataModule 
from model import CPTFoundationModel

from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler


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
    
    # --- Get Training Parameters ---
    training_params = config['training_params']
    learning_rate = training_params['learning_rate']
    num_epochs = training_params['num_epochs']
    mask_ratio = training_params.get('mask_ratio', 0.15) # Get mask_ratio from config

    # --- 2. Data Loading (now much cleaner) ---
    print("Setting up data module...")
    data_module = CPTDataModule(config)
    data_module.setup() # This will preprocess if necessary and setup all splits
    
    # Get DataLoaders for train and validation
    train_loader = data_module.get_dataloader(stage='train', shuffle=True)
    val_loader = data_module.get_dataloader(stage='val', shuffle=False)
    
    print("Data loading complete.")

    # --- 3. Model Initialization ---
    model = CPTFoundationModel(
        num_features=model_params['num_features'],
        model_dim=model_params['model_dim'],
        num_heads=model_params['num_heads'],
        num_layers=model_params['num_layers']
    ).to(device)

    # --- Setup Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    scaler = GradScaler()

    print("Starting training...")
    # --- 4. Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Use tqdm for a progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch, attention_mask in pbar:
                batch = batch.to(device)
                attention_mask = attention_mask.to(device)
                
                # --- Create a corrupted version of the input batch ---
                corrupted_batch = batch.clone()
                prob_mask = torch.rand(batch.shape[:2], device=device)
                masking_condition = (prob_mask < mask_ratio) & (attention_mask == 1)
                
                num_masked = masking_condition.sum().item()
                if num_masked == 0:
                    continue # Skip this batch if no tokens are masked
                
                # Apply the mask. Here we set masked values to 0.0
                # Another option could be a learned [MASK] embedding.
                corrupted_batch[masking_condition] = 0.0

                # --- Forward Pass ---
                optimizer.zero_grad()

                with autocast():
                    # Get the contextual embeddings from the model
                    contextual_embeddings = model(corrupted_batch, attention_mask)
                    
                    # Apply the final projection ONLY to the masked tokens
                    masked_embeddings = contextual_embeddings[masking_condition]
                    predictions = model.output_projection(masked_embeddings)

                    # Calculate loss ONLY on the values that were masked.
                    loss = loss_fn(predictions, batch[masking_condition])
                
                # --- Backward Pass ---
                # 3. Scale the loss and call backward() on the scaled loss. 
                # this is to prevent underflow (gradients becoming too small)
                scaler.scale(loss).backward()
                
                # Clip gradients to prevent them from exploding
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 4. scaler.step() unscales gradients and calls optimizer.step()
                scaler.step(optimizer)
                
                # 5. Update the scale for the next iteration
                scaler.update()
                
                if torch.isfinite(loss):
                    total_loss += loss.item()
                
                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")

        # Save a model checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, model_save_path)

    print(f"\nTraining complete. Final model saved to '{model_save_path}'")

if __name__ == '__main__':
    # Set up argument parser to read the config file path from the command line
    # example: python src/train.py --config configs/PG_dataset.yaml
    DEFAULT_CONFIG_PATH = 'configs/PG_dataset.yaml'
    parser = argparse.ArgumentParser(description="Train a CPT Foundation Model.")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load the YAML configuration file
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration file loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        exit()
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit()
    
    # Start the training process
    train(config)