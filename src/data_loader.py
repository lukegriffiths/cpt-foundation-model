import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class CPTTensorDataset(Dataset):
    """Loads pre-processed CPT tensor files."""
    def __init__(self, processed_dir):
        self.file_paths = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return torch.load(self.file_paths[idx])

def collate_cpts(batch):
    """Pads sequences in a batch to the same length and creates an attention mask."""
    # `batch` is a list of tensors
    
    # Change the padding_value from 0.0 to a large negative number
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=-9999.0) # <-- The change is here
    
    # Create a mask to tell the model which parts are real data (1) vs. padding (0)
    lengths = [len(seq) for seq in batch]
    attention_mask = torch.zeros(padded_batch.shape[:2], dtype=torch.float32)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1.0
        
    return padded_batch, attention_mask