import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)
        Q = self.q_proj(x) # Txmodel_dim x model_dim x model_dim = Txmodel_dim
        K = self.k_proj(x) # Txmodel_dim x model_dim x num_kv_heads * self.head_dim = Tx(num_kv_heads*self.head_dim)
        V = self.v_proj(x) # Txmodel_dim x model_dim x num_kv_heads * self.head_dim = Tx(num_kv_heads*self.head_dim)

        Q_reshaped = Q.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        K_reshaped = K.reshape(B, T, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        V_reshaped = V.reshape(B, T, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        K_expanded = torch.repeat_interleave(K_reshaped, self.num_heads//self.num_kv_heads, dim=-3) # Bxself.num_headsxTxself.head_dim 
        V_expanded = torch.repeat_interleave(V_reshaped, self.num_heads//self.num_kv_heads, dim=-3) # Bxself.num_headsxTxself.head_dim

        d_k = self.head_dim
        scores = Q_reshaped @ K_expanded.transpose(-1, -2)
        scores /= (d_k)**(0.5)
        tril = torch.tril(torch.ones((T, T)))
        scores = scores.masked_fill(tril==0, float('-inf'))
        scores = nn.functional.softmax(scores, dim=-1)
        attention = scores @ V_expanded # Bxself.num_headsxTxself.head_dim
        attention_reshaped = attention.permute(0, 2, 1, 3).reshape(B, T, -1)
        out = self.output_proj(attention_reshaped)
        return torch.round(out, decimals=4)

        
        