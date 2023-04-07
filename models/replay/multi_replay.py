from typing import Dict, Union, List

import numpy as np


class MultiReplay:
    """A simple experience replay buffer for multi agent.

    It inherited codes from spinningup project: https://github.com/openai/spinningup.
    """

    def __init__(
        self,
        agent_num: int,
        obs_dim: Union[int, List[int]],
        act_dim: Union[int, List[int]],
        max_size: int,
        dtype='float32',
    ):
        """Init a replay buffer.

        Args:
            agent_num: Number of agents.

            obs_dim: Dimension of observation space.

            act_dim: Dimension of action space.

            max_size: Maximum size of buffer.

            dtype: Data type of buffer.
        """
        self.agent_num, self.max_size, self.dtype = agent_num, max_size, dtype
        self.obs_dim = obs_dim if isinstance(obs_dim, list) else [obs_dim] * agent_num
        self.act_dim = act_dim if isinstance(act_dim, list) else [act_dim] * agent_num
        self.obs1_bufs = [np.zeros((max_size, self.obs_dim[i]), dtype=dtype) for i in range(agent_num)]
        self.acts_bufs = [np.zeros((max_size, self.act_dim[i]), dtype=dtype) for i in range(agent_num)]
        self.obs2_bufs = [np.zeros((max_size, self.obs_dim[i]), dtype=dtype) for i in range(agent_num)]
        self.rews_buf = np.zeros((agent_num, max_size), dtype=dtype)
        self.term_buf = np.zeros(max_size, dtype=dtype)
        self.ptr, self.size = 0, 0

    def store(
        self,
        obs: List[np.ndarray],
        act: List[np.ndarray],
        next_obs: List[np.ndarray],
        rew: Union[float, List[float]],
        term: bool,
    ):
        """Store experience data.

        Args:
            obs: Observation.

            act: Action.

            next_obs: Next observation.

            rew: Reward.

            term: Whether a terminal state is reached.
        """
        for i in range(self.agent_num):
            self.obs1_bufs[i][self.ptr] = obs[i]
            self.acts_bufs[i][self.ptr] = act[i]
            self.obs2_bufs[i][self.ptr] = next_obs[i]
        self.rews_buf[:, self.ptr] = rew
        self.term_buf[self.ptr] = term
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """Randomly sample a batch of data from buffer.

        Args:
            batch_size: Size of batch.

        Returns:
            Sampled data.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs1=[buf[idxs] for buf in self.obs1_bufs],
            acts=[buf[idxs] for buf in self.acts_bufs],
            obs2=[buf[idxs] for buf in self.obs2_bufs],
            rews=[np.squeeze(rew) for rew in np.split(self.rews_buf[:, idxs], self.agent_num, axis=0)],
            term=self.term_buf[idxs],
        )

    def get(self) -> Dict[str, Union[int, Dict[str, Union[np.ndarray, List[np.ndarray]]]]]:
        """Get the internal state of the replay buffer.

        Returns:
            Internal state of the replay buffer.
        """
        state = {
            'agent_num': self.agent_num,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'max_size': self.max_size,
            'dtype': self.dtype,
            'ptr': self.ptr,
            'size': self.size,
            'data': {
                'obs1_bufs': [buf[:self.size] for buf in self.obs1_bufs],
                'acts_bufs': [buf[:self.size] for buf in self.acts_bufs],
                'obs2_bufs': [buf[:self.size] for buf in self.obs2_bufs],
                'rews_buf': self.rews_buf[:, :self.size],
                'term_buf': self.term_buf[:self.size],
            }
        }
        return state

    def set(self, state: Dict[str, Union[int, Dict[str, Union[np.ndarray, List[np.ndarray]]]]]):
        """Set the internal state of the replay buffer.

        Args:
            state: Internal state of the replay buffer.
        """
        s = state
        self.agent_num, self.max_size, self.dtype = s['agent_num'], s['max_size'], s['dtype']
        self.obs_dim, self.act_dim = s['obs_dim'], s['act_dim']
        self.ptr, self.size = s['ptr'], s['size']

        self.obs1_bufs = [np.zeros((self.max_size, self.obs_dim[i]), dtype=self.dtype) for i in range(self.agent_num)]
        self.acts_bufs = [np.zeros((self.max_size, self.act_dim[i]), dtype=self.dtype) for i in range(self.agent_num)]
        self.obs2_bufs = [np.zeros((self.max_size, self.obs_dim[i]), dtype=self.dtype) for i in range(self.agent_num)]
        self.rews_buf = np.zeros((self.agent_num, self.max_size), dtype=self.dtype)
        self.term_buf = np.zeros(self.max_size, dtype=self.dtype)

        d = s['data']
        for i in range(self.agent_num):
            self.obs1_bufs[i][:self.size] = d['obs1_bufs'][i]
            self.acts_bufs[i][:self.size] = d['acts_bufs'][i]
            self.obs2_bufs[i][:self.size] = d['obs2_bufs'][i]
        self.rews_buf[:, :self.size] = d['rews_buf']
        self.term_buf[:self.size] = d['term_buf']
