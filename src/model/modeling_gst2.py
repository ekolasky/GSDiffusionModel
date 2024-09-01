
import torch
import torch.nn as nn
import torch.optim as optim

class GST(nn.Module):
    """
    Gaussian Splat Transformer
    """
    def __init__(self, input_dim, hidden_size, num_layers, num_heads=3, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, hidden_size)
        self.transformer_blocks = nn.ModuleList(
            [GSTBlock(hidden_size, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.model_dim)
        
    
def GSTBlock(nn.Module):
    """
    GST Transformer Block
    """

    def __init__(self, hidden_size, num_heads, **block_kwargs):
        super().__init__()
        
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.forward = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

        

    def forward(self, x):
        

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Example usage:
# model = VanillaTransformerModel(input_dim=1000, model_dim=512, num_heads=8, num_layers=6, output_dim=10)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

