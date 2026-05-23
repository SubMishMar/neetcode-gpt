import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions = []
        with torch.no_grad():
            for layer in model:
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    # To calculate dead fraction, we need to check if all activations 
                    # are <=0 for a particular column.
                    # i.e. we want to see if the same neuron is always reporting <=0 
                    # for all elements in the training batch.
                    # num of cols is the number of neurons, or the number of output features
                    # num of rows is the number of training samples
                    dead_fraction = round(torch.mean((x <= 0).all(dim=0).float()).item(),4)
                    dead_fractions.append(dead_fraction)
        return dead_fractions
            

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        for dead_fraction in dead_fractions:
            if dead_fraction > 0.5:
                return 'use_leaky_relu'
                
        if dead_fractions[0] > 0.3:
            return 'reinitialize'
        
        increasing_dead_fraction = True
        for index in range(len(dead_fractions)-1):
            diff = dead_fractions[index+1]-dead_fractions[index]
            if diff < 0:
                increasing_dead_fraction = False

        if dead_fractions[-1] > 0 and increasing_dead_fraction:
            return 'reduce_learning_rate'

        if max(dead_fractions) < 0.1:
            return 'healthy'

        return 'healthy'