import numpy as np


def discount_cumsum(x: np.ndarray, discount: float):
    y = np.array(x)
    for i in reversed(range(1, len(x))):
        y[i - 1] += discount * y[i]
    return y
