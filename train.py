import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        B = batch_size
        T = context_length
        optimizer = torch.optim.AdamW(model.parameters(), lr)
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            idx = torch.randint(0, len(data)-context_length, (batch_size, ))
            X = torch.stack([data[id:id+context_length] for id in idx], dim=0)
            Y = torch.stack([data[id+1:id+context_length+1] for id in idx], dim=0)
            logits = model(X) # BxTxC
            _, _, C = logits.shape
            logits = logits.reshape(B*T, C)
            Y = Y.reshape(B*T)
            loss = F.cross_entropy(logits, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return round(loss.item(), 4)

