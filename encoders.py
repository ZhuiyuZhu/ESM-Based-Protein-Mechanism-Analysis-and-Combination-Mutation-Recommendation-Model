import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralFeatureEncoder(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class SimpleGNN(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            d_in = node_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(d_in, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for lin, norm in zip(self.layers, self.norms):
            row, col = edge_index
            neigh = x[col]

            agg = torch.zeros_like(x)
            agg.index_add_(0, row, neigh)
            deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1).unsqueeze(1)
            agg = agg / deg

            out = lin(x + agg)
            out = norm(out)
            out = F.silu(out)
            x = out
            x = self.dropout(x)
        return x


class MutationDeltaEncoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

    def forward(self, h_wt, h_mut, mask):
        # 标记是否需要 squeeze 输出
        squeeze_output = False

        if h_wt.dim() == 2:
            h_wt = h_wt.unsqueeze(0)
            h_mut = h_mut.unsqueeze(0)
            mask = mask.unsqueeze(0)
            squeeze_output = True

        mask_f = mask.unsqueeze(-1).float()
        h_wt_local = (h_wt * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        h_mut_local = (h_mut * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        z = torch.cat([h_wt_local, h_mut_local], dim=-1)
        z = self.net(z)

        # 如果输入原本是单样本，squeeze 掉 batch 维，保证输出是 [hidden_dim]
        if squeeze_output:
            z = z.squeeze(0)

        return z