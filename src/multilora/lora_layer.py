import stk.backend
import stk.backend.sputnik
import torch
import torch.nn as nn
import numpy as np
from typing import List
import stk

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

class MultiLoRALayerSTK(LoRALayer):
    def __init__(self, in_features, out_features, adapter_ids, ranks: List[int], alpha=1.0, dropout=0.0):
        """
        
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ranks = ranks
        self.R = sum(ranks)
        self.scalings = alpha / torch.tensor(ranks, dtype=torch.bfloat16).to('cuda')
        self.dropout = nn.Dropout(p=dropout)
        self.BLOCKS_SIZE = 16
        assert all(rank % self.BLOCKS_SIZE == 0 for rank in ranks)
        self.cumr = np.concat(([0], np.cumsum(ranks)))

        self.A = nn.Parameter(torch.randn(self.R, out_features, device="cuda", dtype=torch.bfloat16) * 0.02)  # Small random init
        self.B = nn.Parameter(torch.zeros(in_features, self.R, device="cuda", dtype=torch.bfloat16))

        self.adapter_ids = adapter_ids

    def forward(self, x):
        init_shape = x.shape
        B, S, H = init_shape

        pad_len = (self.BLOCKS_SIZE - (S % self.BLOCKS_SIZE)) % self.BLOCKS_SIZE
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len), mode='constant', value=0)
            S += pad_len  # Update S to reflect new shape
        
        assert S % self.BLOCKS_SIZE == 0
        assert H % self.BLOCKS_SIZE == 0
        

        # GENERATE MASK MATRIX
        local_ids = self.adapter_ids.to('cpu').kron(
            torch.ones((S // self.BLOCKS_SIZE), dtype=int)
        )
        labels = []
        for id in local_ids.numpy():
            labels.append(list(range(self.cumr[id] // self.BLOCKS_SIZE, self.cumr[id + 1] // self.BLOCKS_SIZE)))

        column_indices = sum(labels, [])
        row_indices = []
        offsets = [0]
        for i, lb in enumerate(labels):
            for l in lb:
                row_indices.append(i)
            offsets.append(len(row_indices))

        column_indices = torch.tensor(column_indices).to('cuda', dtype=torch.int16)
        row_indices = torch.tensor(row_indices).to('cuda', dtype=torch.int16)
        offsets = torch.tensor(offsets).to('cuda', dtype=torch.int32)

        topo = stk.Matrix(
            size=(B * S, self.R), 
            data=torch.ones((len(column_indices), self.BLOCKS_SIZE, self.BLOCKS_SIZE), dtype=torch.float16, device="cuda"),
            row_indices=row_indices,
            column_indices=column_indices,
            offsets=offsets
        )
        topo.validate()

        # Perform calculations
        x_prime = x.view(-1, x.shape[2])
        Bx = stk.ops.sdd(x_prime, self.B, topo)
        result = stk.ops.dsd(Bx, self.A)

        # Undo the padding if needed before returning
        result = result.view((B, S, self.out_features))
        if pad_len > 0:
            result = result[:, :-pad_len, :]

        return result * self.scalings[self.adapter_ids.to('cpu')].view(-1, 1, 1)
    
    

