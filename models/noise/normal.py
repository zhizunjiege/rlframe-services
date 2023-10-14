from typing import Iterable, Optional, Tuple, Union

import numpy as np


class NormalNoise:

    def __init__(
        self,
        mu: Union[float, Iterable[float]] = 0.0,
        sigma: Union[float, Iterable[float]] = 1.0,
        shape: Optional[Union[int, Tuple[int, ...]]] = None,
    ):
        self.mu = mu
        self.sigma = sigma
        self.shape = shape

    def __call__(self):
        return np.random.normal(loc=self.mu, scale=self.sigma, size=self.shape)

    def reset(self):
        ...
