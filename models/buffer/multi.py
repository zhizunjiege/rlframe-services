from typing import Dict, List, Optional, Union

import numpy as np


class MultiAgentBuffer:
    """A simple experience replay buffer for multi agent.

    It inherited codes from spinningup project: https://github.com/openai/spinningup.
    """

    def __init__(
        self,
        number: int,
        obs_dim: Union[int, List[int]],
        act_dim: Union[int, List[int]],
        max_size: int,
        dtype=np.float32,
    ):
        """Init a buffer.

        Args:
            number: Number of agents.
            obs_dim: Dimension of observation space.
            act_dim: Dimension of action space.
            max_size: Maximum size of buffer.
            dtype: Data type of buffer.
        """
        obs_dim = obs_dim if isinstance(obs_dim, list) else [obs_dim] * number
        act_dim = act_dim if isinstance(act_dim, list) else [act_dim] * number
        self.obs1_bufs = [np.zeros((max_size, obs_dim[i]), dtype=dtype) for i in range(number)]
        self.acts_bufs = [np.zeros((max_size, act_dim[i]), dtype=dtype) for i in range(number)]
        self.obs2_bufs = [np.zeros((max_size, obs_dim[i]), dtype=dtype) for i in range(number)]
        self.rews_bufs = [np.zeros((max_size, 1), dtype=dtype) for _ in range(number)]
        self.term_buf = np.zeros((max_size, 1), dtype=dtype)
        self.number, self.max_size = number, max_size
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
        obs: Union[List[np.ndarray], Dict[int, np.ndarray]],
        act: Union[List[np.ndarray], Dict[int, np.ndarray]],
        next_obs: Union[List[np.ndarray], Dict[int, np.ndarray]],
        rew: Union[List[float], Dict[int, float]],
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
        for i in range(self.number):
            self.obs1_bufs[i][ptr] = obs[i]
            self.acts_bufs[i][ptr] = act[i]
            self.obs2_bufs[i][ptr] = next_obs[i]
            self.rews_bufs[i][ptr] = rew[i]
        self.term_buf[ptr] = term
        self.ptr += 1

    def sample(
        self,
        batch_size: int,
        batch_idxs: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """Randomly sample a batch of data from buffer.

        Args:
            batch_size: Size of batch.
            batch_idxs: Indexes of batch.

        Returns:
            Sampled data.
        """
        idxs = np.random.randint(0, self.size, size=batch_size) if batch_idxs is None else batch_idxs
        return dict(
            obs1=[buf[idxs] for buf in self.obs1_bufs],
            acts=[buf[idxs] for buf in self.acts_bufs],
            obs2=[buf[idxs] for buf in self.obs2_bufs],
            rews=[buf[idxs] for buf in self.rews_bufs],
            term=self.term_buf[idxs],
        )

    def get(self) -> Dict[str, Union[int, np.ndarray, List[np.ndarray]]]:
        """Get the internal state of the buffer.

        Returns:
            Internal state of the buffer.
        """
        return dict(
            ptr=self.ptr,
            obs1=[buf[:self.size] for buf in self.obs1_bufs],
            acts=[buf[:self.size] for buf in self.acts_bufs],
            obs2=[buf[:self.size] for buf in self.obs2_bufs],
            rews=[buf[:self.size] for buf in self.rews_bufs],
            term=self.term_buf[:self.size],
        )

    def set(self, state: Dict[str, Union[int, np.ndarray, List[np.ndarray]]]):
        """Set the internal state of the buffer.

        Args:
            state: Internal state of the buffer.
        """
        self.ptr = state['ptr']
        for i in range(self.number):
            self.obs1_bufs[i][:self.size] = state['obs1'][i]
            self.acts_bufs[i][:self.size] = state['acts'][i]
            self.obs2_bufs[i][:self.size] = state['obs2'][i]
            self.rews_bufs[i][:self.size] = state['rews'][i]
        self.term_buf[:self.size] = state['term']
