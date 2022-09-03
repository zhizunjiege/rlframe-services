from typing import Dict, Union

import numpy as np


class SimpleReplay:
    """A simple experience replay buffer.

    It inherited codes from spinningup project: https://github.com/openai/spinningup.
    """

    def __init__(self, obs_dim: int, act_dim: int, max_size: int, dtype=np.float32) -> None:
        """Init a replay buffer.

        Args:
            obs_dim: Dimension of observation space.

            act_dim: Dimension of action space.

            max_size: Maximum size of buffer.

            dtype: Data type of buffer.
        """
        self.obs1_buf = np.zeros([max_size, obs_dim], dtype=dtype)
        self.obs2_buf = np.zeros([max_size, obs_dim], dtype=dtype)
        self.acts_buf = np.zeros([max_size, act_dim], dtype=dtype)
        self.rews_buf = np.zeros(max_size, dtype=dtype)
        self.done_buf = np.zeros(max_size, dtype=dtype)
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def store(self, obs: np.ndarray, act: Union[int, np.ndarray], rew: float, next_obs: np.ndarray, done: bool) -> None:
        """Store experience data.

        Args:
            obs: Observation.

            act: Action.

            rew: Reward.

            next_obs: Next observation.

            done: Indicate whether terminated or not.
        """
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of data from buffer.

        Args:
            batch_size: Size of batch.

        Returns:
            Sampled data.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )
