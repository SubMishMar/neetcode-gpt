import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        # x is 1xi, W1 is hxi, b1 is 1xh
        # z1 is 1xh
        x = np.array(x, dtype=np.float64).reshape(1,-1)
        W1 = np.array(W1, dtype=np.float64)
        b1 = np.array(b1, dtype=np.float64)
        W2 = np.array(W2, dtype=np.float64)
        b2 = np.array(b2, dtype=np.float64)
        y_true = np.array(y_true, dtype=np.float64).reshape(1,-1)

        z1 = np.matmul(x, W1.T) + b1
        dz1_dW1 = x # 1 x i
        dz1_db1 = np.ones_like(b1) # 1 x h

        # a1 is 1xh
        a1 = np.maximum(0, z1)
        da1_dz1 = (a1 > 0).astype(np.float32)# 1xh

        # predictions is 1 x o, W2 is o x h, b2 is 1 x o
        predictions = np.matmul(a1, W2.T) + b2
        dpredictions_dW2 = a1 # 1 x h
        dpredictions_db2 = np.ones_like(b2) # 1 x o
        dpredictions_da1 = W2

        error = predictions - y_true
        loss = np.mean(np.square(predictions - y_true))

        n = len(y_true)
        # dpredictions is 1xo
        dpredictions = 2*error/n
        # dW2 is oxh
        dW2 = dpredictions.T @ dpredictions_dW2
        # db2 is 1xo
        db2 = dpredictions*dpredictions_db2
        # da1 is 1xh
        da1 = dpredictions @ dpredictions_da1
        # dW1 is hxi
        dW1 = (da1*da1_dz1).T @ dz1_dW1
        # db1 is 1xh
        db1 = da1*da1_dz1*dz1_db1

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1[0], 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2[0], 4).tolist()
        }

