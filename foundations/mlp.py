import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:       

    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        h = x
        for weight, bias in zip(weights[:-1], biases[:-1]):
            weight = np.array(weight, dtype=np.float64)
            bias = np.array(bias, dtype=np.float64)
            h = np.matmul(h, weight) + bias
            h = np.maximum(0, h)
        h = np.matmul(h, weights[-1]) + biases[-1]
        return np.round(h, 5)

