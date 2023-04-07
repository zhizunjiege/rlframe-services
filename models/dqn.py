from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from .base import RLModelBase
from .replay.single_replay import SingleReplay


class DQN(RLModelBase):
    """Deep Q-learning Network model."""

    def __init__(
        self,
        training: bool,
        *,
        obs_dim: int = 4,
        act_num: int = 2,
        hidden_layers: List[int] = [64, 64],
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
        dtype: str = 'float32',
    ):
        """Init a DQN model.

        Args:
            training: Whether the model is in training mode.

            obs_dim: Dimension of observation.
            act_num: Number of actions.
            hidden_layers: Units of hidden layers.
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
            dtype: Data type of model.
        """
        super().__init__(training)

        self.obs_dim = obs_dim
        self.act_num = act_num
        self.hidden_layers = hidden_layers
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
        self.dtype = dtype

        if training:
            tf.random.set_seed(seed)
            np.random.seed(seed)

            self.online_net = self.net_builder('online', obs_dim, hidden_layers, act_num)
            self.target_net = self.net_builder('target', obs_dim, hidden_layers, act_num)
            self.update_target()

            self.optimizer = tf.keras.optimizers.Adam(lr)
            self.replay_buffer = SingleReplay(obs_dim, 1, replay_size, dtype=dtype)

            self.log_dir = f'data/logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)
            tf.summary.trace_on(graph=True, profiler=False)

            self._epsilon = epsilon_max
            self._react_steps = 0
            self._train_steps = 0
            self._episode = 0
            self._episode_rewards = 0
            self._graph_exported = False
        else:
            self.online_net = self.net_builder('online', obs_dim, hidden_layers, act_num)

    def __del__(self):
        """Close model."""
        ...

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
                logits = self.online_net(states, training=False)
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
        """Store experience repplay data.

        Args:
            states: States of enviroment.
            actions: Actions of model.
            next_states: Next states of enviroment.
            reward: Reward.
            terminated: Whether a `terminal state` (as defined under the MDP of the task) is reached.
            truncated: Whether a truncation condition outside the scope of the MDP is satisfied.
        """
        self.replay_buffer.store(states, actions, next_states, reward, terminated)

        self._episode_rewards += reward
        if terminated or truncated:
            self._episode += 1
            with self.summary_writer.as_default():
                tf.summary.scalar('episode_rewards', self._episode_rewards, step=self._episode)
            self._episode_rewards = 0

    def train(self):
        """Train model."""
        if self.replay_buffer.size >= self.update_after and self._react_steps % self.update_online_every == 0:
            for _ in range(self.update_online_every):
                batch = self.replay_buffer.sample(self.batch_size)
                loss = self.apply_grads(
                    batch['obs1'],
                    batch['acts'].astype(np.int32).squeeze(axis=1),
                    batch['obs2'],
                    batch['rews'],
                    batch['term'],
                )
                self._train_steps += 1
                with self.summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=self._train_steps)
                    if not self._graph_exported:
                        tf.summary.trace_export(name='model', step=self._train_steps, profiler_outdir=self.log_dir)
                        self._graph_exported = True
                if self._train_steps % self.update_target_every == 0:
                    self.update_target()

    def net_builder(self, name, input_dim, hidden_layers, output_dim):
        inputs = tf.keras.Input(shape=(input_dim,))
        outputs = inputs
        for layer in hidden_layers:
            outputs = tf.keras.layers.Dense(units=layer, activation='relu')(outputs)
        outputs = tf.keras.layers.Dense(units=output_dim, activation='linear')(outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    @tf.function
    def apply_grads(self, states, actions, next_states, rewards, terminated):
        with tf.GradientTape() as tape:
            logits = self.online_net(states, training=True)
            q_values = tf.math.reduce_sum(logits * tf.one_hot(actions, self.act_num), axis=1)
            next_logits = self.target_net(next_states, training=True)
            next_q_values = tf.math.reduce_max(next_logits, axis=1)
            target_q_values = rewards + self.gamma * (1 - terminated) * next_q_values
            td_errors = tf.stop_gradient(target_q_values) - q_values
            loss = tf.math.reduce_mean(tf.math.square(td_errors))
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

    def get_buffer(self) -> Dict[str, Union[int, str, Dict[str, np.ndarray]]]:
        """Get buffer of experience replay.

        Returns:
            Internel state of the simple replay buffer.
        """
        return self.replay_buffer.get()

    def set_buffer(self, buffer: Dict[str, Union[int, str, Dict[str, np.ndarray]]]):
        """Set buffer of experience replay.

        Args:
            buffer: Internel state of the simple replay buffer.
        """
        self.replay_buffer.set(buffer)
        self.replay_size = buffer['max_size']
