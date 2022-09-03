from datetime import datetime
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from .base import RLModelBase
from .replay.simple_replay import SimpleReplay


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

            networks: Networks of model.

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

            self.online_net = networks['online']
            self.target_net = networks['target']
            self.update_target()

            self._epsilon = epsilon_max
            self._react_steps = 0
            self._train_steps = 0
            self._states_dim = self.online_net.layers[0].input_shape[0][1]
            self._actions_num = self.online_net.layers[-1].output_shape[1]

            self.optimizer = tf.keras.optimizers.Adam(lr)
            self.replay_buffer = SimpleReplay(self._states_dim, 1, replay_size, dtype=np.float32)

            self.log_dir = f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)
            # tf.summary.trace_on(graph=True, profiler=True)
            # with self.summary_writer.as_default():
            #     tf.summary.graph(tf.Graph())
        else:
            self.online_net = networks['online']
            self.target_net = None
            self.replay_buffer = None

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
                logits = self.online_net(states, training=False)
                action = np.argmax(logits[0])
            self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)
            self._react_steps += 1
        else:
            states = states[np.newaxis, :]
            logits = self.online_net(states, training=False)
            action = np.argmax(logits[0])
        return int(action)

    def store(self, states: np.ndarray, actions: int, next_states: np.ndarray, reward: float, done: bool) -> None:
        """Store experience repplay data.

        Args:
            states: States of enviroment.

            actions: Actions of model.

            next_states: Next states of enviroment.

            reward: Reward.

            done: Indicating whether terminated or not.
        """
        self.replay_buffer.store(states, actions, reward, next_states, done)

    def train(self) -> None:
        """Train model."""
        if self.replay_buffer.size >= self.update_after and self._react_steps % self.update_online_every == 0:
            for _ in range(self.update_online_every):
                batch = self.replay_buffer.sample(self.batch_size)

                grads = self.calc_grads(
                    batch['obs1'],
                    batch['acts'].astype(np.int32).squeeze(axis=1),
                    batch['obs2'],
                    batch['rews'],
                    batch['done'],
                )

                self.optimizer.apply_gradients(zip(grads, self.online_net.trainable_variables))
                self._train_steps += 1
                if self._train_steps % self.update_target_every == 0:
                    self.update_target()

    @tf.function
    def calc_grads(self, states, actions, next_states, rewards, done):
        with tf.GradientTape() as tape:
            logits = self.online_net(states, training=True)
            q_values = tf.math.reduce_sum(logits * tf.one_hot(actions, self._actions_num), axis=1)
            next_logits = self.target_net(next_states, training=True)
            next_q_values = tf.math.reduce_max(next_logits, axis=1)
            target_q_values = rewards + self.gamma * (1 - done) * next_q_values
            td_errors = tf.stop_gradient(target_q_values) - q_values
            loss = tf.math.reduce_mean(tf.math.square(td_errors))
        grads = tape.gradient(loss, self.online_net.trainable_variables)
        with self.summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=self._train_steps)
            tf.summary.scalar('td_error', tf.math.reduce_mean(tf.math.abs(td_errors)), step=self._train_steps)
        return grads

    def update_target(self):
        self.target_net.set_weights(self.online_net.get_weights())

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get weights of neural networks.

        Returns:
            Weights of `online network` and `target network`(if exists).
        """
        weights = {
            'online': self.online_net.get_weights(),
        }
        if self.training and self.target_net is not None:
            weights['target'] = self.target_net.get_weights()
        return weights

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set weights of neural networks.

        Args:
            weights: Weights of `online network` and `target network`(if exists).
        """
        self.online_net.set_weights(weights['online'])
        if self.training and 'target' in weights:
            self.target_net.set_weights(weights['target'])

    def get_buffer(self) -> SimpleReplay:
        """Get replay buffer of model.

        Returns:
            Replay buffer.
        """
        return self.replay_buffer

    def set_buffer(self, buffer: SimpleReplay) -> None:
        """Set replay buffer of model.

        Args:
            buffer: Replay buffer.
        """
        self.replay_buffer = buffer
