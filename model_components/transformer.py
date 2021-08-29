import torch
import torch.nn as nn

from .residual_attention_block import ResidualAttentionBlock

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

# if __name__ == '__main__':
#     x = torch.randn(20,16,100) # sequence_len, batch, embed_dim
#     with torch.no_grad():
#         y = Transformer(width=100, layers=5, heads=5)(x)
#     print(y.size())
