from typing import Dict, Optional, Union

import numpy as np


class SingleAgentBuffer:
    """A experience replay buffer for single agent.

    It inherited codes from spinningup project: https://github.com/openai/spinningup.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_size: int,
        dtype=np.float32,
    ):
        """Init a buffer.

        Args:
            obs_dim: Dimension of observation space.
            act_dim: Dimension of action space.
            max_size: Maximum size of buffer.
            dtype: Data type of buffer.
        """
        self.obs1_buf = np.zeros((max_size, obs_dim), dtype=dtype)
        self.acts_buf = np.zeros((max_size, act_dim), dtype=dtype)
        self.obs2_buf = np.zeros((max_size, obs_dim), dtype=dtype)
        self.rews_buf = np.zeros((max_size, 1), dtype=dtype)
        self.term_buf = np.zeros((max_size, 1), dtype=dtype)
        self.max_size = max_size
        self.ptr = 0

    @property
    def size(self) -> int:
        """Size of buffer.

        Returns:
            Size of buffer.
        """
        return min(self.ptr, self.max_size)

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
        rew: float,
        term: bool,
    ):
        """Store experience replay data.

        Args:
            obs: Observation.
            act: Action.
            next_obs: Next observation.
            rew: Reward.
            term: Whether a terminal state is reached.
        """
        ptr = self.ptr % self.max_size
        self.obs1_buf[ptr] = obs
        self.acts_buf[ptr] = act
        self.obs2_buf[ptr] = next_obs
        self.rews_buf[ptr] = rew
        self.term_buf[ptr] = term
        self.ptr += 1

    def sample(
        self,
        batch_size: int,
        batch_idxs: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of data from buffer.

        Args:
            batch_size: Size of batch.
            batch_idxs: Indexes of batch.

        Returns:
            Sampled data.
        """
        idxs = np.random.randint(0, self.size, size=batch_size) if batch_idxs is None else batch_idxs
        return dict(
            obs1=self.obs1_buf[idxs],
            acts=self.acts_buf[idxs],
            obs2=self.obs2_buf[idxs],
            rews=self.rews_buf[idxs],
            term=self.term_buf[idxs],
        )

    def get(self) -> Dict[str, Union[int, np.ndarray]]:
        """Get the internal state of the buffer.

        Returns:
            Internal state of the buffer.
        """
        return dict(
            ptr=self.ptr,
            obs1=self.obs1_buf[:self.size],
            acts=self.acts_buf[:self.size],
            obs2=self.obs2_buf[:self.size],
            rews=self.rews_buf[:self.size],
            term=self.term_buf[:self.size],
        )

    def set(self, state: Dict[str, Union[int, np.ndarray]]):
        """Set the internal state of the buffer.

        Args:
            state: Internal state of the buffer.
        """
        self.ptr = state['ptr']
        self.obs1_buf[:self.size] = state['obs1']
        self.acts_buf[:self.size] = state['acts']
        self.obs2_buf[:self.size] = state['obs2']
        self.rews_buf[:self.size] = state['rews']
        self.term_buf[:self.size] = state['term']
