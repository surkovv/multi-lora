import torch
import torch.nn as nn
from typing import List

class LoRALayer(nn.Module):
    def __init__(self):
        super().__init__()


class SingleLoRALayer(LoRALayer):
    def __init__(self, in_features, out_features, rank=8, alpha=1.0, dropout=0.0):
        """
        LoRA layer implementation with trainable A and B matrices.
        
        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
            rank (int): Rank of the low-rank decomposition.
            alpha (float): Scaling factor for LoRA layers.
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.02)  # Small random init
        self.B = nn.Parameter(torch.zeros(rank, in_features))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout((self.A @ (self.B @ x.T)).T) * self.scaling
    

class MultiLoRALayerMaskingHom(LoRALayer):
    def __init__(self, in_features, out_features, adapter_ids, rank=8, alpha=1.0, n_adapters=1, dropout=0.0):
        """
        
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(n_adapters, out_features, rank) * 0.02)  # Small random init
        self.B = nn.Parameter(torch.zeros(n_adapters, rank, in_features))
        self.scaling = alpha / rank
        self.adapter_ids = adapter_ids
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        Bx = torch.einsum("bni, ari -> bnar", x, self.B)
        result = torch.einsum("bnar, aor -> bnao", Bx, self.A)
        return self.dropout(
            result[torch.arange(len(self.adapter_ids)), :, self.adapter_ids, ...]
        ) * self.scaling
    

class MultiLoRALayerMaskingHomEfficient(LoRALayer):
    def __init__(self, in_features, out_features, adapter_ids, rank=8, alpha=1.0, n_adapters=1, dropout=0.0):
        """
        
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(n_adapters, out_features, rank) * 0.02)  # Small random init
        self.B = nn.Parameter(torch.zeros(n_adapters, rank, in_features))
        self.scaling = alpha / rank
        self.adapter_ids = adapter_ids
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        B_request = self.B[self.adapter_ids]
        A_request = self.A[self.adapter_ids]
        Bx = torch.einsum("bni, bri -> bnr", x, B_request)
        result = torch.einsum("bnr, bor -> bno", Bx, A_request)
        return self.dropout(result) * self.scaling


class MultiLoRALayerMasking(LoRALayer):
    def __init__(self, in_features, out_features, adapter_ids, ranks: List[int], alpha=1.0, dropout=0.0):
        """
        
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ranks = ranks
        self.loras = nn.ModuleList([
            SingleLoRALayer(in_features, out_features, rank, alpha, dropout)
            for rank in ranks
        ])
        self.adapter_ids = adapter_ids

    def forward(self, x):
        result = torch.zeros(x.shape[:-1] + (self.out_features,), dtype=x.dtype, device=x.device)
        for adapter_id in range(len(self.ranks)):
            x_adapter = x.masked_select(self.adapter_ids.view(-1, 1, 1) == adapter_id)
            if x_adapter.numel() == 0:
                continue
            x_adapter = x_adapter.view(-1, x.shape[-1])
            result_adapter = self.loras[adapter_id](x_adapter)
            result.masked_scatter_(self.adapter_ids.view(-1, 1, 1) == adapter_id, result_adapter)

        return result


