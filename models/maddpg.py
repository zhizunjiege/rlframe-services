from typing import Dict, Iterable, List, Literal, Optional, Union

import numpy as np
import tensorflow as tf

from .base import RLModelBase
from .buffer import MultiAgentBuffer
from .core import MLPModel
from .noise import NormalNoise, OrnsteinUhlenbeckNoise


class MADDPG(RLModelBase):

    def __init__(
        self,
        training: bool,
        *,
        number: int,
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
        noise_type: Literal['normal', 'ou'] = 'normal',
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
        """Init a MADDPG model.

        Args:
            training: whether model is used for `train` or `infer`.

            number: Number of agents.
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
            noise_type: Type of noise, `normal` or `ou`.
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

        self.number = number
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

            self.actor_list = []
            self.actor_target_list = []
            self.actor_optimizer_list = []
            self.critic_list = []
            self.critic_target_list = []
            self.critic_optimizer_list = []
            self.noise_list = []

            critic_inputs = [obs_dim * number, act_dim * number]
            for i in range(number):
                actor = MLPModel(f'actor_{i}', True, obs_dim, hidden_layers_actor, 'relu', act_dim, 'tanh')
                self.actor_list.append(actor)
                actor_target = MLPModel(f'actor_target_{i}', False, obs_dim, hidden_layers_actor, 'relu', act_dim, 'tanh')
                self.actor_target_list.append(actor_target)
                self.actor_optimizer_list.append(tf.keras.optimizers.Adam(self.lr_actor))
                self.update_target_weights(self.actor_list[i].weights, self.actor_target_list[i].weights, 1)

                critic = MLPModel(f'critic_{i}', True, critic_inputs, hidden_layers_critic, 'relu', 1, 'linear')
                self.critic_list.append(critic)
                critic_target = MLPModel(f'critic_target_{i}', False, critic_inputs, hidden_layers_critic, 'relu', 1, 'linear')
                self.critic_target_list.append(critic_target)
                self.critic_optimizer_list.append(tf.keras.optimizers.Adam(self.lr_critic))
                self.update_target_weights(self.critic_list[i].weights, self.critic_target_list[i].weights, 1)

                if noise_type == 'normal':
                    self.noise_list.append(NormalNoise(sigma=noise_sigma, shape=(act_dim,)))
                elif noise_type == 'ou':
                    self.noise_list.append(
                        OrnsteinUhlenbeckNoise(
                            sigma=noise_sigma,
                            theta=noise_theta,
                            dt=noise_dt,
                            shape=(act_dim,),
                        ))

            self.buffer = MultiAgentBuffer(number, obs_dim, act_dim, buffer_size)

            self._noise_level = noise_max
            self._react_steps = 0
            self._train_steps = 0
        else:
            for i in range(number):
                actor = MLPModel(f'actor_{i}', False, obs_dim, hidden_layers_actor, 'relu', act_dim, 'tanh')
                self.actor_list.append(actor)

    def react(self, states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Get action.

        Args:
            states: Dict of key for agent index and value for states of enviroment.

        Returns:
            Dict of key for agent index and value for actions of model.
        """
        actions = {}
        for i in range(self.number):
            s = states[i][np.newaxis, :]
            logits = self.actor_list[i](s, training=self.training)
            actions[i] = np.array(logits[0])
        if self.training:
            for i in range(self.number):
                noise = self._noise_level * self.noise_list[i]()
                actions[i] += noise
                # actions[i] = np.clip(actions[i], -1, 1)
            self._react_steps += 1
            self._noise_level = max(self.noise_min, self._noise_level * self.noise_decay)
        return actions

    def store(
        self,
        states: Dict[int, np.ndarray],
        actions: Dict[int, np.ndarray],
        next_states: Dict[int, np.ndarray],
        reward: Dict[int, float],
        terminated: bool,
        truncated: bool,
    ):
        """Store experience replay data.

        Args:
            states: Dict of key for agent index and value for states of enviroment.
            actions: Dict of key for agent index and value for actions of model.
            next_states: Dict of key for agent index and value for next states of enviroment.
            reward: Dict of key for agent index and value for reward of enviroment.
            terminated: Whether a `terminal state` (as defined under the MDP of the task) is reached.
            truncated: Whether a truncation condition outside the scope of the MDP is satisfied.
        """
        if self.training:
            self.buffer.store(states, actions, next_states, reward, terminated)

            if terminated or truncated:
                for i in range(self.number):
                    self.noise_list[i].reset()

    def train(self):
        """Train model.

        Returns:
            Losses of actor and critic.
        """
        losses = {}
        if self.training and self.buffer.size >= self.update_after and self._react_steps % self.update_every == 0:
            for _ in range(self.update_every):
                batch = self.buffer.sample(self.batch_size)
                loss_actors, loss_critics = self.apply_grads(
                    batch['obs1'],
                    batch['acts'],
                    batch['obs2'],
                    batch['rews'],
                    batch['term'],
                )
                self._train_steps += 1
                for i in range(self.number):
                    losses.setdefault(f'loss_actor_agent_{i}', []).append(float(loss_actors[i]))
                    losses.setdefault(f'loss_critic_agent_{i}', []).append(float(loss_critics[i]))
        return losses

    @tf.function
    def update_target_weights(self, weights, target_weights, tau):
        [a.assign(a * (1 - tau) + b * tau) for a, b in zip(target_weights, weights)]

    @tf.function
    def apply_grads(self, obs, act, next_obs, rew, term):
        loss_actors = tf.TensorArray(tf.float32, size=self.number)
        loss_critics = tf.TensorArray(tf.float32, size=self.number)

        all_obs = tf.concat(obs, axis=1)
        all_act = tf.concat(act, axis=1)
        all_next_obs = tf.concat(next_obs, axis=1)
        all_next_act = tf.concat([self.actor_target_list[i](next_obs[i]) for i in range(self.number)], axis=1)
        for i in range(self.number):
            with tf.GradientTape() as tape1:
                q_pred = self.critic_list[i]([all_obs, all_act])
                next_q_pred = self.critic_target_list[i]([all_next_obs, all_next_act])
                q_target = rew[i] + self.gamma * next_q_pred * (1 - term)
                td_error = tf.stop_gradient(q_target) - q_pred
                loss_critic = tf.reduce_mean(tf.square(td_error))
                critic_gradients = tape1.gradient(loss_critic, self.critic_list[i].trainable_variables)
                self.critic_optimizer_list[i].apply_gradients(zip(critic_gradients, self.critic_list[i].trainable_variables))
                loss_critics = loss_critics.write(i, loss_critic)

            with tf.GradientTape() as tape2:
                all_act_sub = tf.concat(act[:i] + [self.actor_list[i](obs[i])] + act[i + 1:], axis=1)
                loss_actor = -tf.reduce_mean(self.critic_list[i]([all_obs, all_act_sub]))
                actor_gradients = tape2.gradient(loss_actor, self.actor_list[i].trainable_variables)
                self.actor_optimizer_list[i].apply_gradients(zip(actor_gradients, self.actor_list[i].trainable_variables))
                loss_actors = loss_actors.write(i, loss_actor)

        for i in range(self.number):
            self.update_target_weights(self.actor_list[i].weights, self.actor_target_list[i].weights, self.tau)
            self.update_target_weights(self.critic_list[i].weights, self.critic_target_list[i].weights, self.tau)

        return loss_actors.stack(), loss_critics.stack()

    def get_weights(self) -> Dict[str, List[List[np.ndarray]]]:
        """Get weights of neural networks.

        Returns:
            Weights of `actor` and `actor_target/critic/critic_target`(if exists).
        """
        weights = {
            'actor': [self.actor_list[i].get_weights() for i in range(self.number)],
        }
        if self.training:
            weights['critic'] = [self.critic_list[i].get_weights() for i in range(self.number)]
            weights['actor_target'] = [self.actor_target_list[i].get_weights() for i in range(self.number)]
            weights['critic_target'] = [self.critic_target_list[i].get_weights() for i in range(self.number)]
        return weights

    def set_weights(self, weights: Dict[str, List[List[np.ndarray]]]):
        """Set weights of neural networks.

        Args:
            weights: Weights of `actor` and `actor_target/critic/critic_target`(if exists).
        """
        for i in range(self.number):
            if 'actor' in weights:
                self.actor_list[i].set_weights(weights['actor'][i])
        if self.training:
            for i in range(self.number):
                if 'critic' in weights:
                    self.critic_list[i].set_weights(weights['critic'][i])
                if 'actor_target' in weights:
                    self.actor_target_list[i].set_weights(weights['actor_target'][i])
                if 'critic_target' in weights:
                    self.critic_target_list[i].set_weights(weights['critic_target'][i])

    def get_buffer(self) -> Optional[Dict[str, Union[int, np.ndarray, List[np.ndarray]]]]:
        """Get the internal state of the buffer.

        Returns:
            Internel state of the buffer.
        """
        if self.training:
            return self.buffer.get()
        else:
            return None

    def set_buffer(self, buffer: Optional[Dict[str, Union[int, np.ndarray, List[np.ndarray]]]]):
        """Set the internal state of the buffer.

        Args:
            buffer: Internel state of the buffer.
        """
        if self.training:
            if buffer is not None:
                self.buffer.set(buffer)
