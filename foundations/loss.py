import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        eps = 1e-7
        y_pred_safe = y_pred + eps
        term1 = np.dot(y_true, np.log(y_pred_safe))
        term2 = np.dot(1.0-y_true, np.log(1.0-y_pred_safe))
        n = len(y_pred)
        return -np.round((term1+term2)/n, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_pred_safe = y_pred + 1e-7
        ce_arr = y_true*np.log(y_pred_safe)
        summation = np.sum(ce_arr)
        n_samples, _ = y_true.shape
        return -np.round(summation/n_samples, 4)

