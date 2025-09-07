import torch.nn as nn

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
        # The mask should be boolean with True indicating positions to ignore
        mask = ~src_key_padding_mask.bool()
        
        x = self.input_projection(src)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        predictions = self.output_projection(x)
        return predictions