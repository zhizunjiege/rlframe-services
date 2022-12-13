from typing import Dict, Union

import numpy as np


class SimpleReplay:
    """A simple experience replay buffer.

    It inherited codes from spinningup project: https://github.com/openai/spinningup.
    """

    def __init__(self, obs_dim: int, act_dim: int, max_size: int, dtype='float32') -> None:
        """Init a replay buffer.

        Args:
            obs_dim: Dimension of observation space.

            act_dim: Dimension of action space.

            max_size: Maximum size of buffer.

            dtype: Data type of buffer.
        """
        self.obs_dim, self.act_dim, self.max_size, self.dtype = obs_dim, act_dim, max_size, dtype
        self.obs1_buf = np.zeros([max_size, obs_dim], dtype=dtype)
        self.obs2_buf = np.zeros([max_size, obs_dim], dtype=dtype)
        self.acts_buf = np.zeros([max_size, act_dim], dtype=dtype)
        self.rews_buf = np.zeros(max_size, dtype=dtype)
        self.done_buf = np.zeros(max_size, dtype=dtype)
        self.ptr, self.size = 0, 0

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

    def get(self) -> Dict[str, Union[int, Dict[str, np.ndarray]]]:
        """Get the internal state of the replay buffer.

        Returns:
            Internal state of the replay buffer.
        """
        state = {
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'max_size': self.max_size,
            'dtype': self.dtype,
            'ptr': self.ptr,
            'size': self.size,
            'data': {
                'obs1_buf': self.obs1_buf[:self.size],
                'obs2_buf': self.obs2_buf[:self.size],
                'acts_buf': self.acts_buf[:self.size],
                'rews_buf': self.rews_buf[:self.size],
                'done_buf': self.done_buf[:self.size],
            }
        }
        return state

    def set(self, state: Dict[str, Union[int, Dict[str, np.ndarray]]]) -> None:
        """Set the internal state of the replay buffer.

        Args:
            state: Internal state of the replay buffer.
        """
        s = state
        self.obs_dim, self.act_dim, self.max_size, self.dtype = s['obs_dim'], s['act_dim'], s['max_size'], s['dtype']
        self.ptr, self.size = s['ptr'], s['size']

        d = s['data']
        self.obs1_buf = np.zeros([self.max_size, self.obs_dim], dtype=self.dtype)
        self.obs2_buf = np.zeros([self.max_size, self.obs_dim], dtype=self.dtype)
        self.acts_buf = np.zeros([self.max_size, self.act_dim], dtype=self.dtype)
        self.rews_buf = np.zeros(self.max_size, dtype=self.dtype)
        self.done_buf = np.zeros(self.max_size, dtype=self.dtype)
        self.obs1_buf[:self.size] = d['obs1_buf']
        self.obs2_buf[:self.size] = d['obs2_buf']
        self.acts_buf[:self.size] = d['acts_buf']
        self.rews_buf[:self.size] = d['rews_buf']
        self.done_buf[:self.size] = d['done_buf']
