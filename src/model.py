import torch.nn as nn
import math

class CPTFoundationModel(nn.Module):
    def __init__(self, num_features, model_dim=128, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(num_features, model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(model_dim, num_features)
        
    def forward(self, src, src_key_padding_mask):
        # The transformer's `src_key_padding_mask` expects True for padding tokens
        # and False for real tokens. Our mask has 1 for real tokens and 0 for padding.
        # We need to invert it.
        padding_mask = (src_key_padding_mask == 0)
        
        x = self.input_projection(src) * math.sqrt(self.model_dim)
        x = self.pos_encoder(x)
        
        # Get the contextual embeddings from the transformer
        contextual_embeddings = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Return the embeddings. The projection will be handled in the training loop.
        return contextual_embeddings