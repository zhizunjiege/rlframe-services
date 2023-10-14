from typing import Iterable, Optional, Tuple, Union

import numpy as np


class OrnsteinUhlenbeckNoise:

    def __init__(
        self,
        mu: Union[float, Iterable[float]] = 0.0,
        sigma: Union[float, Iterable[float]] = 0.2,
        theta: Union[float, Iterable[float]] = 0.15,
        dt=0.01,
        shape: Optional[Union[int, Tuple[int, ...]]] = None,
    ):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.shape = shape

        self.reset()

        self.shape = self.x.shape

    def __call__(self):
        self.x = self.x + self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.shape)
        return self.x

    def reset(self):
        self.x = np.random.normal(loc=self.mu, scale=self.sigma, size=self.shape)
