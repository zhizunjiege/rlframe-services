from typing import Dict, Iterable, List, Literal, Optional, Union

import numpy as np
import tensorflow as tf

from .base import RLModelBase
from .buffer import SingleAgentBuffer
from .core import MLPModel
from .noise import NormalNoise, OrnsteinUhlenbeckNoise


class DDPG(RLModelBase):

    def __init__(
        self,
        training: bool,
        *,
        obs_dim: int,
        act_dim: int,
        hidden_layers_actor: List[int] = [64, 64],
        hidden_layers_critic: List[int] = [64, 64],
        lr_actor=0.0001,
        lr_critic=0.001,
        gamma=0.99,
        tau=0.001,
        buffer_size=1000000,
        batch_size=64,
        noise_type: Literal['ou', 'normal'] = 'ou',
        noise_sigma: Union[float, Iterable[float]] = 0.2,
        noise_theta: Union[float, Iterable[float]] = 0.15,
        noise_dt=0.01,
        noise_max=1.0,
        noise_min=1.0,
        noise_decay=1.0,
        update_after=64,
        update_every=1,
        seed: Optional[int] = None,
    ):
        """Init a DDPG model.

        Args:
            training: whether model is used for `train` or `infer`.

            obs_dim: Dimension of observation.
            act_dim: Dimension of actions.
            hidden_layers_actor: Units of actor hidden layers.
            hidden_layers_critic: Units of critic hidden layers.
            lr_actor: Learning rate of actor network.
            lr_critic: Learning rate of critic network.
            gamma: Discount factor.
            tau: Soft update factor.
            buffer_size: Maximum size of buffer.
            batch_size: Size of batch.
            noise_type: Type of noise, `ou` or `normal`.
            noise_sigma: Sigma of noise.
            noise_theta: Theta of noise, `ou` only.
            noise_dt: Delta time of noise, `ou` only.
            noise_max: Maximum value of noise.
            noise_min: Minimum value of noise.
            noise_decay: Decay rate of noise.
                Note: Noise decayed exponentially, so always between 0 and 1.
            update_after: Number of env interactions to collect before starting to do gradient descent updates.
                Note: Ensures buffer is full enough for useful updates.
            update_every: Number of env interactions that should elapse between gradient descent updates.
                Note: Regardless of how long you wait between updates, the ratio of env steps to gradient steps is locked to 1.
            seed: Seed for random number generators.
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
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.noise_type = noise_type
        self.noise_sigma = noise_sigma
        self.noise_theta = noise_theta
        self.noise_dt = noise_dt
        self.noise_max = noise_max
        self.noise_min = noise_min
        self.noise_decay = noise_decay
        self.update_after = update_after
        self.update_every = update_every
        self.seed = seed

        if training:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)

            self.actor = MLPModel('actor', True, hidden_layers_actor, 'relu', act_dim, 'tanh')
            self.actor_target = MLPModel('actor_target', False, hidden_layers_actor, 'relu', act_dim, 'tanh')
            self.actor_optimizer = tf.keras.optimizers.Adam(lr_actor)
            self.update_target_weights(self.actor.weights, self.actor_target.weights, 1)

            self.critic = MLPModel('critic', True, hidden_layers_critic, 'relu', 1, 'linear')
            self.critic_target = MLPModel('critic_target', False, hidden_layers_critic, 'relu', 1, 'linear')
            self.critic_optimizer = tf.keras.optimizers.Adam(lr_critic)
            self.update_target_weights(self.critic.weights, self.critic_target.weights, 1)

            self.buffer = SingleAgentBuffer(obs_dim, act_dim, buffer_size)

            if noise_type == 'normal':
                self.noise = NormalNoise(sigma=noise_sigma, shape=(act_dim,))
            elif noise_type == 'ou':
                self.noise = OrnsteinUhlenbeckNoise(
                    sigma=noise_sigma,
                    theta=noise_theta,
                    dt=noise_dt,
                    shape=(act_dim,),
                )

            self._noise_level = noise_max
            self._react_steps = 0
            self._train_steps = 0
        else:
            self.actor = MLPModel('actor', False, hidden_layers_actor, 'relu', act_dim, 'tanh')

    def react(self, states: np.ndarray) -> np.ndarray:
        """Get action.

        Args:
            states: States of enviroment.

        Returns:
            Action.
        """
        states = states[np.newaxis, :]
        logits = self.actor(states, training=self.training)
        actions = np.array(logits[0])
        if self.training:
            noise = self._noise_level * self.noise()
            actions += noise
            # actions = np.clip(actions, -1, 1)
            self._react_steps += 1
            self._noise_level = max(self.noise_min, self._noise_level * self.noise_decay)
        return actions

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
        if self.training:
            self.buffer.store(states, actions, next_states, reward, terminated)

            if terminated or truncated:
                self.noise.reset()

    def train(self):
        """Train model.

        Returns:
            Losses of actor and critic.
        """
        losses = {}
        if self.training and self.buffer.size >= self.update_after and self._react_steps % self.update_every == 0:
            for _ in range(self.update_every):
                batch = self.buffer.sample(self.batch_size)
                loss_actor, loss_critic = self.apply_grads(
                    batch['obs1'],
                    batch['acts'],
                    batch['obs2'],
                    batch['rews'],
                    batch['term'],
                )
                self._train_steps += 1
                losses.setdefault('loss_actor', []).append(float(loss_actor))
                losses.setdefault('loss_critic', []).append(float(loss_critic))
        return losses

    @tf.function
    def update_target_weights(self, weights, target_weights, tau):
        [a.assign(a * (1 - tau) + b * tau) for a, b in zip(target_weights, weights)]

    @tf.function
    def apply_grads(self, obs, act, next_obs, rew, term):
        with tf.GradientTape() as tape1:
            q_pred = self.critic([obs, act])
            next_act = self.actor_target(next_obs)
            next_q_pred = self.critic_target([next_obs, next_act])
            q_target = rew + self.gamma * next_q_pred * (1 - term)
            td_error = tf.stop_gradient(q_target) - q_pred
            loss_critic = tf.reduce_mean(tf.square(td_error))
        critic_grads = tape1.gradient(loss_critic, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape2:
            act_sub = self.actor(obs)
            loss_actor = -tf.reduce_mean(self.critic([obs, act_sub]))
        actor_grads = tape2.gradient(loss_actor, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self.update_target_weights(self.actor.weights, self.actor_target.weights, self.tau)
        self.update_target_weights(self.critic.weights, self.critic_target.weights, self.tau)
        return loss_actor, loss_critic

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
