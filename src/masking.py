import torch
import torch.nn as nn
import numpy as np


def create_span_mask(batch_shape, attention_mask, mask_ratio=0.15, mean_span_length=5, device='cpu'):
    """
    Creates a mask with contiguous spans rather than individual points.
    
    Args:
        batch_shape: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len) - 1 for real tokens, 0 for padding
        mask_ratio: Proportion of tokens to mask overall
        mean_span_length: Average length of masked spans
        device: Device to create tensors on
    
    Returns:
        masking_condition: Boolean tensor of positions to mask
    """
    batch_size, seq_len = batch_shape
    masking_condition = torch.zeros(batch_shape, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        # Get valid positions for this sequence
        valid_positions = (attention_mask[b] == 1).nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            continue
            
        num_valid = len(valid_positions)
        num_to_mask = int(num_valid * mask_ratio)
        
        masked_count = 0
        attempts = 0
        max_attempts = 100
        
        while masked_count < num_to_mask and attempts < max_attempts:
            # Sample a span length from geometric distribution
            span_length = min(
                np.random.geometric(1.0 / mean_span_length),
                num_to_mask - masked_count,
                num_valid - masked_count
            )
            
            # Sample a valid starting position
            max_start = num_valid - span_length
            if max_start <= 0:
                break
                
            start_idx = np.random.randint(0, max_start + 1)
            
            # Check if this span overlaps with already masked positions
            positions_to_mask = valid_positions[start_idx:start_idx + span_length]
            if not masking_condition[b, positions_to_mask].any():
                masking_condition[b, positions_to_mask] = True
                masked_count += span_length
            
            attempts += 1
    
    return masking_condition


def create_block_mask(batch_shape, attention_mask, mask_ratio=0.15, block_size=10, device='cpu'):
    """
    Creates a mask with fixed-size blocks.
    
    Args:
        batch_shape: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        mask_ratio: Proportion of tokens to mask
        block_size: Size of each masked block
        device: Device to create tensors on
    
    Returns:
        masking_condition: Boolean tensor of positions to mask
    """
    batch_size, seq_len = batch_shape
    masking_condition = torch.zeros(batch_shape, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        valid_positions = (attention_mask[b] == 1).nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            continue
            
        num_valid = len(valid_positions)
        num_blocks = int((num_valid * mask_ratio) / block_size)
        
        if num_blocks == 0:
            num_blocks = 1
            
        for _ in range(num_blocks):
            # Sample a starting position for the block
            max_start = num_valid - block_size
            if max_start <= 0:
                # If sequence is shorter than block size, mask what we can
                masking_condition[b, valid_positions] = True
                break
                
            start_idx = np.random.randint(0, max_start + 1)
            positions_to_mask = valid_positions[start_idx:start_idx + block_size]
            masking_condition[b, positions_to_mask] = True
    
    return masking_condition


def apply_mask_to_input(batch, masking_condition, mask_type='noise', mask_value=None, attention_mask=None):
    """
    Applies masking to the input batch using different strategies.
    
    Args:
        batch: Input tensor (batch_size, seq_len, num_features)
        masking_condition: Boolean tensor of positions to mask (batch_size, seq_len)
        mask_type: Type of masking - 'noise', 'zero', 'mean', or 'learnable'
        mask_value: Optional learned mask embedding (for 'learnable' type)
        attention_mask: Optional attention mask to identify valid positions (batch_size, seq_len)
    
    Returns:
        corrupted_batch: Masked input tensor
    """
    corrupted_batch = batch.clone()
    
    if mask_type == 'noise':
        # Replace with random Gaussian noise
        # Scale noise based on the feature statistics
        
        # Expand mask to match batch dimensions (add feature dimension)
        mask_expanded = masking_condition.unsqueeze(-1).expand_as(batch)
        
        # Compute std from non-masked positions
        if attention_mask is not None:
            # Use only valid, non-masked positions
            valid_mask = (attention_mask == 1) & (~masking_condition)
            valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(batch)
            
            if valid_mask_expanded.any():
                batch_std = batch[valid_mask_expanded].std()
            else:
                batch_std = 1.0  # Fallback if all positions are masked
        else:
            # Use all non-masked positions
            if (~mask_expanded).any():
                batch_std = batch[~mask_expanded].std()
            else:
                batch_std = 1.0  # Fallback if all positions are masked
        
        # Generate noise scaled by the standard deviation
        noise = torch.randn_like(batch) * batch_std * 0.1
        
        # Apply noise only to masked positions
        corrupted_batch = torch.where(
            mask_expanded,
            noise,
            corrupted_batch
        )
    
    elif mask_type == 'zero':
        # Replace with zeros - expand mask to match batch dimensions
        mask_expanded = masking_condition.unsqueeze(-1).expand_as(batch)
        corrupted_batch[mask_expanded] = 0.0
    
    elif mask_type == 'mean':
        # Replace with the mean of each feature
        for feat_idx in range(batch.shape[-1]):
            # Compute mean for this feature from non-masked positions
            feat_data = batch[:, :, feat_idx]
            
            if attention_mask is not None:
                # Use only valid, non-masked positions
                valid_mask = (attention_mask == 1) & (~masking_condition)
                if valid_mask.any():
                    feat_mean = feat_data[valid_mask].mean()
                else:
                    feat_mean = 0.0
            else:
                # Use all non-masked positions
                if (~masking_condition).any():
                    feat_mean = feat_data[~masking_condition].mean()
                else:
                    feat_mean = 0.0
            
            # Apply mean to masked positions for this feature
            corrupted_batch[:, :, feat_idx] = torch.where(
                masking_condition,
                torch.full_like(masking_condition, feat_mean, dtype=torch.float),
                batch[:, :, feat_idx]
            )
    
    elif mask_type == 'learnable':
        # Use a learnable mask token
        if mask_value is None:
            raise ValueError("mask_value must be provided for learnable mask type")
        mask_expanded = masking_condition.unsqueeze(-1).expand_as(batch)
        corrupted_batch = torch.where(
            mask_expanded,
            mask_value.expand_as(batch),
            corrupted_batch
        )
    
    return corrupted_batch