import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # Register 'pe' as a buffer, so it's part of the model's state but not a parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding to the input tensor
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CPTFoundationModel(nn.Module):
    def __init__(self, num_features, model_dim=128, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        
        # Input projection layer
        self.input_projection = nn.Linear(num_features, model_dim)
        
        # Positional Encoder
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection layer
        self.output_projection = nn.Linear(model_dim, num_features)

    def forward(self, src, src_key_padding_mask):
        # The transformer's `src_key_padding_mask` expects True for padding tokens
        # and False for real tokens. Our mask has 1 for real tokens and 0 for padding.
        padding_mask = (src_key_padding_mask == 0)
        
        # Project, add positional encoding, then run through transformer
        x = self.input_projection(src) * math.sqrt(self.model_dim)
        x = self.pos_encoder(x)
        contextual_embeddings = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        return contextual_embeddings