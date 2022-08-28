from datetime import datetime
from typing import Dict, Optional, Union

import numpy as np
import tensorflow as tf

from .base import RLModelBase


class ReplayBuffer:
    """A simple experience replay buffer.

    It inherited codes from spinningup project: https://github.com/openai/spinningup.
    """

    def __init__(self, obs_dim: int, act_dim: int, max_size: int) -> None:
        """Init a replay buffer.

        Args:
            obs_dim: Dimension of observation space.

            act_dim: Dimension of action space.

            max_size: Maximum size of buffer.
        """
        self.obs1_buf = np.zeros([max_size, obs_dim])
        self.obs2_buf = np.zeros([max_size, obs_dim])
        self.acts_buf = np.zeros([max_size, act_dim])
        self.rews_buf = np.zeros(max_size)
        self.done_buf = np.zeros(max_size)
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


class DQN(RLModelBase):
    """Deep Q-learning Network model."""

    def __init__(
        self,
        training: bool,
        networks: Dict[str, tf.keras.Model],
        *,
        lr: float = 0.001,
        gamma: float = 0.95,
        replay_size: int = 1000000,
        batch_size: int = 32,
        epsilon_max: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.9,
        start_steps: int = 0,
        update_after: int = 32,
        update_online_every: int = 1,
        update_target_every: int = 200,
        seed: Optional[int] = None,
    ):
        """Init a DQN model.

        Args:
            training: Whether the model is in training mode.

            lr: Learning rate.

            gamma: Discount factor.

            replay_size: Maximum size of replay buffer.

            batch_size: Size of batch.

            epsilon_max: Maximum value of epsilon.

            epsilon_min: Minimum value of epsilon.

            epsilon_decay: Decay rate of epsilon.
                Note: Epsilon decayed exponentially, so always between 0 and 1.

            start_steps: Number of steps for uniform-random action selection before running real policy.
                Note: Helps exploration.

            update_after: Number of env interactions to collect before starting to do gradient descent updates.
                Note: Ensures replay buffer is full enough for useful updates.

            update_online_every: Number of env interactions that should elapse between gradient descent updates.
                Note: Regardless of how long you wait between updates, the ratio of env steps to gradient steps is locked to 1.

            update_target_every: Number of env interactions that should elapse between target network updates.

            seed: Seed for random number generators.
        """
        super().__init__(training)

        self.lr = lr
        self.gamma = gamma
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_online_every = update_online_every
        self.update_target_every = update_target_every
        self.seed = seed

        if training:
            tf.random.set_seed(seed)
            np.random.seed(seed)

            self.__online_net = networks['online']
            self.__target_net = networks['target']
            self.__update_target()

            self._epsilon = epsilon_max
            self._react_steps = 0
            self._train_steps = 0
            self._states_dim = self.__online_net.layers[0].input_shape[1]
            self._actions_num = self.__online_net.layers[-1].output_shape[1]

            self.__optimizer = tf.keras.optimizers.Adam(lr)
            self.__replay_buffer = ReplayBuffer(self._states_dim, 1, replay_size)

            log_dir = f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            self.__summary_writer = tf.summary.create_file_writer(log_dir)
            tf.summary.trace_on(graph=True, profiler=True)
            with self.__summary_writer.as_default():
                tf.summary.graph(tf.Graph())
        else:
            self.__online_net = networks['online']
            self.__target_net = None
            self.__replay_buffer = None

    def react(self, states: np.ndarray) -> int:
        """Get action.

        Args:
            states: States of enviroment.

        Returns:
            Action.
        """
        if self.training:
            if self._react_steps < self.start_steps or np.random.random() < self._epsilon:
                action = np.random.randint(0, self._actions_num)
            else:
                states = states[np.newaxis, :]
                logits = self.__online_net(states, training=False)
                action = np.argmax(logits[0])
            self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)
            self._react_steps += 1
        else:
            states = states[np.newaxis, :]
            logits = self.__online_net(states, training=False)
            action = np.argmax(logits[0])
        return action

    def store(self, states: np.ndarray, actions: int, next_states: np.ndarray, reward: float, done: bool) -> None:
        """Store experience repplay data.

        Args:
            states: States of enviroment.

            actions: Actions of model.

            next_states: Next states of enviroment.

            reward: Reward.

            done: Indicating whether terminated or not.
        """
        self.__replay_buffer.store(states, actions, next_states, reward, done)

    def train(self) -> None:
        """Train model."""
        if self.__replay_buffer.size >= self.update_after and self._react_steps % self.update_online_every == 0:
            with self.__summary_writer.as_default():
                for _ in range(self.update_online_every):
                    batch = self.__replay_buffer.sample(self.batch_size)
                    with tf.GradientTape() as tape:
                        logits = self.__online_net(batch['obs1'], training=True)
                        q_values = tf.reduce_sum(logits * tf.one_hot(batch['acts'].squeeze(axis=1), self._actions_num), axis=1)
                        next_logits = self.__target_net(batch['obs2'], training=True)
                        next_q_values = tf.reduce_max(next_logits, axis=1)
                        target_q_values = batch['rews'] + self.gamma * (1 - batch['done']) * next_q_values
                        td_errors = tf.stop_gradient(target_q_values) - q_values
                        loss = tf.reduce_mean(tf.math.square(td_errors))
                    grads = tape.gradient(loss, self.__online_net.trainable_variables)
                    self.__optimizer.apply_gradients(zip(grads, self.__online_net.trainable_variables))
                    self._train_steps += 1
                    if self._train_steps % self.update_target_every == 0:
                        self.__update_target()

                    tf.summary.scalar('loss', loss, step=self._train_steps)
                    tf.summary.scalar('td_error', tf.reduce_mean(tf.abs(td_errors)), step=self._train_steps)

    def __update_target(self):
        self.__target_net.set_weights(self.__online_net.get_weights())

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get weights of neural networks.

        Returns:
            Weights of `online network` and `target network`(if exists).
        """
        weights = {
            'online': self.__online_net.get_weights(),
        }
        if self.training and self.__target_net is not None:
            weights['target'] = self.__target_net.get_weights()
        return weights

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set weights of neural networks.

        Args:
            weights: Weights of `online network` and `target network`(if exists).
        """
        self.__online_net.set_weights(weights['online'])
        if self.training and 'target' in weights:
            self.__target_net.set_weights(weights['target'])

    def get_buffer(self) -> ReplayBuffer:
        """Get replay buffer of model.

        Returns:
            Replay buffer.
        """
        return self.__replay_buffer

    def set_buffer(self, buffer: ReplayBuffer) -> None:
        """Set replay buffer of model.

        Args:
            buffer: Replay buffer.
        """
        self.__replay_buffer = buffer
