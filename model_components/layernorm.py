import torch
import torch.nn as nn

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

if __name__ == '__main__':
    x = torch.randn(16,75,200)
    with torch.no_grad():
        y = LayerNorm(200)(x)
    print(y.size())