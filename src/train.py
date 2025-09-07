import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import your custom classes
from data_loader import CPTTensorDataset, collate_cpts
from model import CPTFoundationModel

# --- Configuration ---
PROCESSED_DIR = 'data/processed'
MODEL_SAVE_PATH = 'cpt_foundation_model.pth'
NUM_FEATURES = 25 # IMPORTANT: Update this to 3 + number of your soil classes
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
MASK_RATIO = 0.15

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Initialize Dataset and DataLoader
dataset = CPTTensorDataset(processed_dir=PROCESSED_DIR)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_cpts)

# 2. Initialize Model, Optimizer, and Loss Function
model = CPTFoundationModel(num_features=NUM_FEATURES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# --- Training Loop ---
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    for batch, attention_mask in data_loader:
        batch = batch.to(device)
        attention_mask = attention_mask.to(device)
        
        # --- Masked Modeling Task ---
        # Create a corrupted version of the input
        corrupted_batch = batch.clone()
        # Probability mask for masking tokens
        prob_mask = torch.rand(batch.shape[:2], device=device)
        # Determine which tokens to mask (must be real data, not padding)
        masking_condition = (prob_mask < MASK_RATIO) & (attention_mask == 1)
        corrupted_batch[masking_condition] = 0.0 # Or a special [MASK] token value

        # --- Forward Pass ---
        optimizer.zero_grad()
        predictions = model(corrupted_batch, attention_mask)
        
        # --- Loss Calculation ---
        # IMPORTANT: Calculate loss ONLY on the values that were masked
        loss = loss_fn(predictions[masking_condition], batch[masking_condition])
        
        # --- Backward Pass ---
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.6f}")

    # Save a checkpoint after each epoch
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("Training complete. Model saved.")