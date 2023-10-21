from typing import Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from .base import RLModelBase
from .buffer import SingleAgentBuffer
from .core import MLPModel


class DoubleDQN(RLModelBase):
    """Double Deep Q-learning Network model."""

    def __init__(
        self,
        training: bool,
        *,
        obs_dim: int,
        act_num: int,
        hidden_layers: List[int] = [64, 64],
        lr=0.001,
        gamma=0.99,
        buffer_size=1000000,
        batch_size=64,
        epsilon_max=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.9,
        start_steps=0,
        update_after=64,
        update_online_every=1,
        update_target_every=200,
        seed: Optional[int] = None,
    ):
        """Init a Double DQN model.

        Args:
            training: whether model is used for `train` or `infer`.

            obs_dim: Dimension of observation.
            act_num: Number of actions.
            hidden_layers: Units of hidden layers.
            lr: Learning rate.
            gamma: Discount factor.
            buffer_size: Maximum size of buffer.
            batch_size: Size of batch.
            epsilon_max: Maximum value of epsilon.
            epsilon_min: Minimum value of epsilon.
            epsilon_decay: Decay rate of epsilon.
                Note: Epsilon decayed exponentially, so always between 0 and 1.
            start_steps: Number of steps for uniform-random action selection before running real policy.
                Note: Helps exploration.
            update_after: Number of env interactions to collect before starting to do gradient descent updates.
                Note: Ensures buffer is full enough for useful updates.
            update_online_every: Number of env interactions that should elapse between gradient descent updates.
                Note: Regardless of how long you wait between updates, the ratio of env steps to gradient steps is locked to 1.
            update_target_every: Number of gradient updations that should elapse between target network updates.
            seed: Seed for random number generators.
        """
        super().__init__(training)

        self.obs_dim = obs_dim
        self.act_num = act_num
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = buffer_size
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

            self.online_net = MLPModel('online', True, obs_dim, hidden_layers, 'relu', act_num, 'linear')
            self.target_net = MLPModel('target', False, obs_dim, hidden_layers, 'relu', act_num, 'linear')
            self.update_target()

            self.optimizer = tf.keras.optimizers.Adam(lr)
            self.buffer = SingleAgentBuffer(obs_dim, 1, buffer_size)

            self._epsilon = epsilon_max
            self._react_steps = 0
            self._train_steps = 0
        else:
            self.online_net = MLPModel('online', False, obs_dim, hidden_layers, 'relu', act_num, 'linear')

    def react(self, states: np.ndarray) -> int:
        """Get action.

        Args:
            states: States of enviroment.

        Returns:
            Action.
        """
        if self.training:
            if self._react_steps < self.start_steps or np.random.random() < self._epsilon:
                action = np.random.randint(0, self.act_num)
            else:
                states = states[np.newaxis, :]
                logits = self.online_net(states, training=True)
                action = np.argmax(logits[0])
            self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)
            self._react_steps += 1
        else:
            states = states[np.newaxis, :]
            logits = self.online_net(states, training=False)
            action = np.argmax(logits[0])
        return int(action)

    def store(
        self,
        states: np.ndarray,
        actions: int,
        next_states: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ):
        """Store experience replay data.

        Args:
            states: States of enviroment.
            actions: Actions of model.
            next_states: Next states of enviroment.
            reward: Reward of enviroment.
            terminated: Whether a `terminal state` (as defined under the MDP of the task) is reached.
            truncated: Whether a truncation condition outside the scope of the MDP is satisfied.
        """
        if self.training:
            self.buffer.store(states, actions, next_states, reward, terminated)

    def train(self) -> Dict[str, List[float]]:
        """Train model.

        Returns:
            Losses.
        """
        losses = {}
        if self.training and self.buffer.size >= self.update_after and self._react_steps % self.update_online_every == 0:
            for _ in range(self.update_online_every):
                batch = self.buffer.sample(self.batch_size)
                loss = self.apply_grads(
                    batch['obs1'],
                    batch['acts'].astype(np.int32).squeeze(axis=1),
                    batch['obs2'],
                    batch['rews'],
                    batch['term'],
                )
                self._train_steps += 1
                if self._train_steps % self.update_target_every == 0:
                    self.update_target()
                losses.setdefault('loss', []).append(float(loss))
        return losses

    @tf.function
    def apply_grads(self, states, actions, next_states, rewards, terminated):
        with tf.GradientTape() as tape:
            logits = self.online_net(states, training=True)
            q_values = tf.reduce_sum(logits * tf.one_hot(actions, self.act_num), axis=1, keepdims=True)
            qmax_acts = tf.argmax(self.online_net(next_states, training=True), axis=1)
            next_logits = self.target_net(next_states, training=True)
            next_q_values = tf.reduce_sum(next_logits * tf.one_hot(qmax_acts, self.act_num), axis=1, keepdims=True)
            target_q_values = rewards + self.gamma * next_q_values * (1 - terminated)
            td_error = tf.stop_gradient(target_q_values) - q_values
            loss = tf.reduce_mean(tf.square(td_error))
        grads = tape.gradient(loss, self.online_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_net.trainable_variables))
        return loss

    def update_target(self):
        self.target_net.set_weights(self.online_net.get_weights())

    def get_weights(self) -> Dict[str, List[np.ndarray]]:
        """Get weights of neural networks.

        Returns:
            Weights of `online network` and `target network`(if exists).
        """
        weights = {
            'online': self.online_net.get_weights(),
        }
        if self.training:
            weights['target'] = self.target_net.get_weights()
        return weights

    def set_weights(self, weights: Dict[str, List[np.ndarray]]):
        """Set weights of neural networks.

        Args:
            weights: Weights of `online network` and `target network`(if exists).
        """
        if 'online' in weights:
            self.online_net.set_weights(weights['online'])
        if self.training:
            if 'target' in weights:
                self.target_net.set_weights(weights['target'])

    def get_buffer(self) -> Optional[Dict[str, Union[int, np.ndarray]]]:
        """Get the internal state of the buffer.

        Returns:
            Internel state of the buffer.
        """
        if self.training:
            return self.buffer.get()
        else:
            return None

    def set_buffer(self, buffer: Optional[Dict[str, Union[int, np.ndarray]]]):
        """Set the internal state of the buffer.

        Args:
            buffer: Internel state of the buffer.
        """
        if self.training:
            if buffer is not None:
                self.buffer.set(buffer)
