import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        batch_size, context_size, _ = embedded.shape
        # 1. Project input through K, Q, V linear layers
        k = self.key(embedded)
        q = self.query(embedded)
        v = self.value(embedded)
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        scores = q@k.transpose(-2, -1)
        scores /= attention_dim**0.5
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        tril = torch.tril(torch.ones(context_size, context_size))
        scores = scores.masked_fill(tril==0, float('-inf'))
        # 4. Apply softmax(dim=2) to masked scores
        scores = nn.functional.softmax(scores, dim=2)
        # 5. Return (scores @ V) rounded to 4 decimal places
        out = scores@v
        return torch.round(out, decimals=4)
