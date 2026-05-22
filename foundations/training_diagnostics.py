import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        records = []
        with torch.no_grad():
            for index, layer in enumerate(model):
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    mean = torch.round(torch.mean(x), decimals = 4)
                    std = torch.round(torch.std(x), decimals = 4)
                    # To calculate dead fraction, we need to check if all activations 
                    # are <=0 for a particular column.
                    # i.e. we want to see if the same neuron is always reporting <=0 
                    # for all elements in the training batch.
                    # num of cols is the number of neurons, or the number of output features
                    # num of rows is the number of training samples
                    dead_fraction = round(torch.mean((x <= 0).all(dim=0).float()).item(),4)
                    record = {'mean' : mean.item(), 'std' : std.item(), 'dead_fraction': dead_fraction}
                    records.append(record)
        return records


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        predictions = model(x)
        loss = nn.MSELoss()(predictions, y)
        loss.backward()
        records = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                mean = round(layer.weight.grad.mean().item(), 4)
                std = round(layer.weight.grad.std().item(), 4)
                norm = round(layer.weight.grad.norm().item(), 4)
                record = {'mean' : mean, 'std' : std, 'norm': norm}
                records.append(record)
        return records


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        activation_std = []
        for activation_stat, gradient_stat in zip(activation_stats, gradient_stats):
            activation_std.append(activation_stat['std'])
            if activation_stat['dead_fraction'] > 0.5:
                return 'dead_neurons'
            elif gradient_stat['norm'] > 1000:
                return 'exploding_gradients'
            elif gradient_stat['norm'] < 1e-5:
                return 'vanishing_gradients'
            else:
                if any(std < 0.1 for std in activation_std):
                    return 'vanishing_gradients'
                elif any(std > 10. for std in activation_std):
                    return 'exploding_gradients'
        return 'healthy'
