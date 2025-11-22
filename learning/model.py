import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # 不更新梯度

    def forward(self, x: torch.Tensor):
        # x: (Batch Size,Seq Len, d_model)
        x = x + self.pe[:x.size(1), :]  # 按序
        return x

def causal_mask(T, device):
    # True=屏蔽未来
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

class FluorToBinaryCausal(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=8, ff=512, dropout=0.1, max_len=200):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(1, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(-1)  # (B, T) -> (B, T, 1)
        h = self.in_proj(x) * math.sqrt(self.d_model)
        h = self.pos(h)
        T = h.size(1)
        mask = causal_mask(T, h.device)
        h = self.enc(h, mask=mask)
        h = self.norm(h)
        logits = self.out(h).squeeze(-1)
        return logits

if __name__ == "__main__":
    # Test the Model
    batch_size = 4
    d_model = 256
    nhead = 8
    num_layers = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    model = FluorToBinaryCausal(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
    print("The parameters of the model:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    x = torch.randn(batch_size, 100, 1).to(device)
    print("The shape of data:", x.shape)

    out = model(x)
    print("Output shape:", out.shape)  # (batch_size, seq_len)

    # Calculate the loss
    target = torch.randint(0, 2, (batch_size, 100)).float().to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(out, target)
    print("Loss:", loss.item())
