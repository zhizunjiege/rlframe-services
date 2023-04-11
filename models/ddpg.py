from datetime import datetime
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from .base import RLModelBase
from .replay.single_replay import SingleReplay


class NormalNoise:

    def __init__(
        self,
        mu: Union[float, Iterable[float]] = 0.0,
        sigma: Union[float, Iterable[float]] = 1.0,
        shape: Optional[Union[int, Tuple[int, ...]]] = None,
    ):
        self.mu = mu
        self.sigma = sigma
        self.shape = shape

    def __call__(self):
        return np.random.normal(loc=self.mu, scale=self.sigma, size=self.shape)


class OrnsteinUhlenbeckNoise:

    def __init__(
        self,
        mu: Union[float, Iterable[float]] = 0.0,
        sigma: Union[float, Iterable[float]] = 0.2,
        theta: Union[float, Iterable[float]] = 0.15,
        dt: float = 0.01,
        shape: Optional[Union[int, Tuple[int, ...]]] = None,
    ):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt

        self.x = np.random.normal(loc=mu, scale=sigma, size=shape)

        self.shape = self.x.shape

    def __call__(self):
        self.x = self.x + self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.shape)
        return self.x


class DDPG(RLModelBase):

    def __init__(
        self,
        training: bool,
        *,
        obs_dim: int = 4,
        act_dim: int = 2,
        hidden_layers_actor: List[int] = [64, 64],
        hidden_layers_critic: List[int] = [64, 64],
        lr_actor: float = 0.0001,
        lr_critic: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.001,
        replay_size: int = 1000000,
        batch_size: int = 64,
        noise_type: Literal['normal', 'ou'] = 'ou',
        noise_sigma: Union[float, Iterable[float]] = 0.2,
        noise_theta: Union[float, Iterable[float]] = 0.15,
        noise_dt: float = 0.01,
        noise_max: float = 1.0,
        noise_min: float = 1.0,
        noise_decay: float = 1.0,
        update_after: int = 64,
        update_online_every: int = 1,
        seed: Optional[int] = None,
        dtype: str = 'float32',
    ):
        """Init a DDPG model.

        Args:
            training: Whether the model is in training mode.

            obs_dim: Dimension of observation.
            act_dim: Dimension of actions.
            hidden_layers_actor: Units of actor hidden layers.
            hidden_layers_critic: Units of critic hidden layers.
            lr_actor: Learning rate of actor network.
            lr_critic: Learning rate of critic network.
            gamma: Discount factor.
            tau: Soft update factor.
            replay_size: Maximum size of replay buffer.
            batch_size: Size of batch.
            noise_type: Type of noise, `normal` or `ou`.
            noise_sigma: Sigma of noise.
            noise_theta: Theta of noise, `ou` only.
            noise_dt: Delta time of noise, `ou` only.
            noise_max: Maximum value of noise.
            noise_min: Minimum value of noise.
            noise_decay: Decay rate of noise.
                Note: Noise decayed exponentially, so always between 0 and 1.
            update_after: Number of env interactions to collect before starting to do gradient descent updates.
                Note: Ensures replay buffer is full enough for useful updates.
            update_online_every: Number of env interactions that should elapse between gradient descent updates.
                Note: Regardless of how long you wait between updates, the ratio of env steps to gradient steps is locked to 1.
            seed: Seed for random number generators.
            dtype: Data type of model.
        """
        super().__init__(training)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers_actor = hidden_layers_actor
        self.hidden_layers_critic = hidden_layers_critic
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.noise_type = noise_type
        self.noise_sigma = noise_sigma
        self.noise_theta = noise_theta
        self.noise_dt = noise_dt
        self.noise_max = noise_max
        self.noise_min = noise_min
        self.noise_decay = noise_decay
        self.update_after = update_after
        self.update_online_every = update_online_every
        self.seed = seed
        self.dtype = dtype

        if training:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)

            self.actor = self.actor_net_builder('actor', obs_dim, hidden_layers_actor, act_dim)
            self.actor_target = self.actor_net_builder('actor_target', obs_dim, hidden_layers_actor, act_dim)
            self.actor_optimizer = tf.keras.optimizers.Adam(lr_actor)
            self.update_target_weights(self.actor.weights, self.actor_target.weights, 1)

            self.critic = self.critic_net_builder('critic', obs_dim + act_dim, hidden_layers_critic, 1)
            self.critic_target = self.critic_net_builder('critic_target', obs_dim + act_dim, hidden_layers_critic, 1)
            self.critic_optimizer = tf.keras.optimizers.Adam(lr_critic)
            self.update_target_weights(self.critic.weights, self.critic_target.weights, 1)

            self.replay_buffer = SingleReplay(obs_dim, act_dim, replay_size, dtype=dtype)

            if noise_type == 'normal':
                self.noise = NormalNoise(sigma=noise_sigma, shape=(act_dim,))
            else:
                self.noise = OrnsteinUhlenbeckNoise(sigma=noise_sigma, theta=noise_theta, dt=noise_dt, shape=(act_dim,))

            self.log_dir = f'data/logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)
            tf.summary.trace_on(graph=True, profiler=False)

            self._noise_level = noise_max
            self._react_steps = 0
            self._train_steps = 0
            self._episode = 0
            self._episode_rewards = 0
            self._graph_exported = False
        else:
            self.actor = self.actor_net_builder('actor', obs_dim, hidden_layers_actor, act_dim)

    def __del__(self):
        """Close model."""
        ...

    def react(self, states: np.ndarray) -> np.ndarray:
        """Get action.

        Args:
            states: States of enviroment.

        Returns:
            Action.
        """
        states = states[np.newaxis, :]
        logits = self.actor(states, training=False)
        actions = logits[0]
        if self.training:
            actions += self._noise_level * self.noise()
            self._react_steps += 1
            self._noise_level = max(self.noise_min, self._noise_level * self.noise_decay)
        return np.clip(actions, -1, 1)

    def store(
        self,
        states: np.ndarray,
        actions: np.ndarray,
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
                actor_loss, critic_loss = self.apply_grads(
                    batch['obs1'],
                    batch['acts'],
                    batch['obs2'],
                    batch['rews'],
                    batch['term'],
                )
                self._train_steps += 1
                with self.summary_writer.as_default():
                    tf.summary.scalar('actor_loss', actor_loss, step=self._train_steps)
                    tf.summary.scalar('critic_loss', critic_loss, step=self._train_steps)
                    if not self._graph_exported:
                        tf.summary.trace_export(name='model', step=self._train_steps, profiler_outdir=self.log_dir)
                        self._graph_exported = True

    def actor_net_builder(self, name, input_dim, hidden_layers, output_dim):
        inputs = tf.keras.Input(shape=(input_dim,))
        outputs = inputs
        for layer in hidden_layers:
            outputs = tf.keras.layers.Dense(units=layer, activation='relu')(outputs)
        outputs = tf.keras.layers.Dense(units=output_dim, activation='tanh')(outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    def critic_net_builder(self, name, input_dim, hidden_layers, output_dim):
        inputs = tf.keras.Input(shape=(input_dim,))
        outputs = inputs
        for layer in hidden_layers:
            outputs = tf.keras.layers.Dense(units=layer, activation='relu')(outputs)
        outputs = tf.keras.layers.Dense(units=output_dim, activation='linear')(outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    @tf.function
    def update_target_weights(self, weights, target_weights, tau):
        [a.assign(a * (1 - tau) + b * tau) for a, b in zip(target_weights, weights)]

    @tf.function
    def apply_grads(self, obs, act, next_obs, rew, term):
        with tf.GradientTape() as tape:
            next_action_target = self.actor_target(next_obs)
            next_q_target = self.critic_target(tf.concat((next_obs, next_action_target), axis=1))
            target_q = rew + self.gamma * next_q_target * (1 - term)
            current_q = self.critic(tf.concat((obs, act), axis=1))
            td_errors = tf.stop_gradient(target_q) - current_q
            critic_loss = tf.reduce_mean(tf.math.square(td_errors))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            sample_action = self.actor(obs)
            actor_loss = -tf.reduce_mean(self.critic(tf.concat((obs, sample_action), axis=1)))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self.update_target_weights(self.actor.weights, self.actor_target.weights, self.tau)
        self.update_target_weights(self.critic.weights, self.critic_target.weights, self.tau)
        return actor_loss, critic_loss

    def get_weights(self) -> Dict[str, List[np.ndarray]]:
        """Get weights of neural networks.

        Returns:
            Weights of `actor` and `actor_target/critic/critic_target`(if exists).
        """
        weights = {
            'actor': self.actor.get_weights(),
        }
        if self.training:
            weights['critic'] = self.critic.get_weights()
            weights['actor_target'] = self.actor_target.get_weights()
            weights['critic_target'] = self.critic_target.get_weights()
        return weights

    def set_weights(self, weights: Dict[str, List[np.ndarray]]):
        """Set weights of neural networks.

        Args:
            weights: Weights of `actor` and `actor_target/critic/critic_target`(if exists).
        """
        if 'actor' in weights:
            self.actor.set_weights(weights['actor'])
        if self.training:
            if 'critic' in weights:
                self.critic.set_weights(weights['critic'])
            if 'actor_target' in weights:
                self.actor_target.set_weights(weights['actor_target'])
            if 'critic_target' in weights:
                self.critic_target.set_weights(weights['critic_target'])

    def get_buffer(self) -> Dict[str, Union[int, str, Dict[str, np.ndarray]]]:
        """Get buffer of experience replay.

        Returns:
            Internel state of the replay buffer.
        """
        return self.replay_buffer.get()

    def set_buffer(self, buffer: Dict[str, Union[int, str, Dict[str, np.ndarray]]]):
        """Set buffer of experience replay.

        Args:
            buffer: Internel state of the replay buffer.
        """
        self.replay_buffer.set(buffer)
        self.replay_size = buffer['max_size']
