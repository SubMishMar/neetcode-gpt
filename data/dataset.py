import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        tokens = raw_dataset.split()
        idx = torch.randint(0, len(tokens)-context_length, (batch_size, ))
        X = [["" for _ in range(context_length)] for _ in range(batch_size)]
        Y = [["" for _ in range(context_length)] for _ in range(batch_size)]
        for i, id in enumerate(idx):
            X[i] = tokens[id:id+context_length]
            Y[i] = tokens[id+1:id+context_length+1]
        return (X, Y)

