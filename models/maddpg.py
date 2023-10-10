from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from .base import RLModelBase
from .replay.multi_replay import MultiReplay


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

    def reset(self):
        ...


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
        self.shape = shape

        self.reset()

        self.shape = self.x.shape

    def __call__(self):
        self.x = self.x + self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.shape)
        return self.x

    def reset(self):
        self.x = np.random.normal(loc=self.mu, scale=self.sigma, size=self.shape)


class MADDPG(RLModelBase):

    def __init__(
        self,
        training: bool,
        *,
        agent_num: int = 2,
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
        dtype: str = 'float32',
        seed: Optional[int] = None,
    ):
        """Init a MADDPG model.

        Args:
            training: whether model is used for `train` or `infer`.

            agent_num: Number of agents.
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
            dtype: Data type of model.
            seed: Seed for random number generators.
        """
        super().__init__(training)

        self.agent_num = agent_num
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
        self.dtype = dtype
        self.seed = seed

        if training:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)

            self.actor_list = []
            self.actor_target_list = []
            self.critic_list = []
            self.critic_target_list = []
            self.noise_list = []
            for i in range(agent_num):
                self.actor_list.append(self.actor_net_builder(f'actor_{i}', trainable=True))
                self.actor_target_list.append(self.actor_net_builder(f'actor_target_{i}', trainable=False))
                self.update_target_weights(self.actor_list[i].weights, self.actor_target_list[i].weights, 1)

                self.critic_list.append(self.critic_net_builder(f'critic_{i}', trainable=True))
                self.critic_target_list.append(self.critic_net_builder(f'critic_target_{i}', trainable=False))
                self.update_target_weights(self.critic_list[i].weights, self.critic_target_list[i].weights, 1)

                if noise_type == 'normal':
                    self.noise_list.append(NormalNoise(sigma=noise_sigma, shape=(act_dim,)))
                else:
                    self.noise_list.append(
                        OrnsteinUhlenbeckNoise(sigma=noise_sigma, theta=noise_theta, dt=noise_dt, shape=(act_dim,)))

            self.replay_buffer = MultiReplay(agent_num, obs_dim, act_dim, replay_size, dtype=dtype)

            self._noise_level = noise_max
            self._react_steps = 0
            self._train_steps = 0
        else:
            for i in range(agent_num):
                self.actor_list.append(self.actor_net_builder(f'actor_{i}', trainable=False))

    def react(self, states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Get action.

        Args:
            states: Dict of key for agent index and value for states of enviroment.

        Returns:
            Dict of key for agent index and value for actions of model.
        """
        action_n = {}
        if self.training:
            for i in range(self.agent_num):
                s = states[i][np.newaxis, :]
                logits = self.actor_list[i](s, training=False)
                noise = self._noise_level * self.noise_list[i]()
                action = logits[0] + noise
                action = np.clip(action, -1, 1)
                action_n[i] = action
            self._react_steps += 1
            self._noise_level = max(self.noise_min, self._noise_level * self.noise_decay)
        else:
            for i in range(self.agent_num):
                s = states[i][np.newaxis, :]
                logits = self.actor_list[i](s, training=False)
                action = logits[0]
                action_n[i] = np.squeeze(action, axis=0)
        return action_n

    def sorted_values(self, d):
        return [v for _, v in sorted(d.items())]

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
            self.replay_buffer.store(
                self.sorted_values(states),
                self.sorted_values(actions),
                self.sorted_values(next_states),
                self.sorted_values(reward),
                terminated,
            )

            if terminated or truncated:
                for i in range(self.agent_num):
                    self.noise_list[i].reset()

    def train(self):
        """Train model.

        Returns:
            Losses of actor and critic.
        """
        losses = {}
        if self.training and self.replay_buffer.size >= self.update_after and self._react_steps % self.update_online_every == 0:
            for _ in range(self.update_online_every):
                batch = self.replay_buffer.sample(self.batch_size)
                loss_actors, loss_critics = self.apply_gradients(
                    batch['obs1'],
                    batch['acts'],
                    batch['obs2'],
                    batch['rews'],
                    batch['term'],
                )
                self._train_steps += 1
                for i in range(self.agent_num):
                    losses.setdefault(f'loss_actor_agent_{i}', []).append(loss_actors[i])
                    losses.setdefault(f'loss_critic_agent_{i}', []).append(loss_critics[i])
        return losses

    def actor_net_builder(self, name, trainable):
        inputs = tf.keras.Input(shape=(self.obs_dim,))
        outputs = inputs
        for layer in self.hidden_layers_actor:
            outputs = tf.keras.layers.Dense(units=layer, activation='relu')(outputs)
        outputs = tf.keras.layers.Dense(units=self.act_dim, activation='tanh')(outputs)
        actor_model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name, trainable=trainable)
        actor_model.compile(optimizer=tf.keras.optimizers.Adam(self.lr_actor))
        return actor_model

    def critic_net_builder(self, name, trainable):
        input_magent_s = tf.keras.Input(shape=(self.agent_num * self.obs_dim,))
        input_magent_a = tf.keras.Input(shape=(self.agent_num * self.act_dim,))
        outputs = tf.concat([input_magent_s, input_magent_a], axis=-1)
        for layer in self.hidden_layers_critic:
            outputs = tf.keras.layers.Dense(units=layer, activation='relu')(outputs)
        outputs = tf.keras.layers.Dense(units=1, activation='linear')(outputs)
        critic_model = tf.keras.Model(inputs=[input_magent_s, input_magent_a], outputs=outputs, name=name, trainable=trainable)
        critic_model.compile(optimizer=tf.keras.optimizers.Adam(self.lr_critic))
        return critic_model

    @tf.function
    def update_target_weights(self, weights, target_weights, tau):
        [a.assign(a * (1 - tau) + b * tau) for a, b in zip(target_weights, weights)]

    @tf.function
    def apply_gradients(self, obs, act, next_obs, rew, term):
        n = self.agent_num
        all_obs = tf.concat(obs, axis=-1)
        all_act = tf.concat(act, axis=-1)
        all_next_obs = tf.concat(next_obs, axis=-1)

        loss_actors = tf.TensorArray(tf.float32, size=n)
        loss_critics = tf.TensorArray(tf.float32, size=n)
        with tf.GradientTape(persistent=True) as tape:
            all_action = tf.concat([self.actor_list[i](obs[i]) for i in range(n)], axis=-1)
            all_next_action = tf.concat([self.actor_target_list[i](next_obs[i]) for i in range(n)], axis=-1)
            for i in range(n):
                actor_pred = self.actor_list[i]
                critic_pred = self.critic_list[i]
                critic_target = self.critic_target_list[i]
                q_pred_critic = critic_pred([all_obs, all_act])
                q_target_critic = rew[i] + self.gamma * critic_target([all_next_obs, all_next_action]) * (1 - term)
                loss_critic = tf.keras.losses.mse(tf.stop_gradient(q_target_critic), q_pred_critic)
                loss_critic = tf.reduce_mean(loss_critic)
                critic_gradients = tape.gradient(loss_critic, critic_pred.trainable_variables)
                critic_pred.optimizer.apply_gradients(zip(critic_gradients, critic_pred.trainable_variables))
                loss_critics = loss_critics.write(i, loss_critic)

                q_pred = critic_pred([all_obs, all_action])
                loss_actor = -tf.math.reduce_mean(q_pred)
                actor_gradients = tape.gradient(loss_actor, actor_pred.trainable_variables)
                actor_pred.optimizer.apply_gradients(zip(actor_gradients, actor_pred.trainable_variables))
                loss_actors = loss_actors.write(i, loss_actor)

        for i in range(n):
            self.update_target_weights(self.actor_list[i].weights, self.actor_target_list[i].weights, self.tau)
            self.update_target_weights(self.critic_list[i].weights, self.critic_target_list[i].weights, self.tau)

        return loss_actors.stack(), loss_critics.stack()

    def get_weights(self) -> Dict[str, List[List[np.ndarray]]]:
        """Get weights of neural networks.

        Returns:
            Weights of `actor` and `actor_target/critic/critic_target`(if exists).
        """
        weights = {
            'actor': [self.actor_list[i].get_weights() for i in range(self.agent_num)],
        }
        if self.training:
            weights['critic'] = [self.critic_list[i].get_weights() for i in range(self.agent_num)]
            weights['actor_target'] = [self.actor_target_list[i].get_weights() for i in range(self.agent_num)]
            weights['critic_target'] = [self.critic_target_list[i].get_weights() for i in range(self.agent_num)]
        return weights

    def set_weights(self, weights: Dict[str, List[List[np.ndarray]]]):
        """Set weights of neural networks.

        Args:
            weights: Weights of `actor` and `actor_target/critic/critic_target`(if exists).
        """
        for i in range(self.agent_num):
            self.actor_list[i].set_weights(weights['actor'][i])
        if self.training:
            for i in range(self.agent_num):
                if 'critic' in weights:
                    self.critic_list[i].set_weights(weights['critic'][i])
                if 'actor_target' in weights:
                    self.actor_target_list[i].set_weights(weights['actor_target'][i])
                if 'critic_target' in weights:
                    self.critic_target_list[i].set_weights(weights['critic_target'][i])

    def get_buffer(self) -> Dict[str, Union[int, str, Dict[str, Union[np.ndarray, List[np.ndarray]]]]]:
        """Get buffer of experience replay.

        Returns:
            Internel state of the replay buffer.
        """
        if self.training:
            return self.replay_buffer.get()
        else:
            return None

    def set_buffer(self, buffer: Dict[str, Union[int, str, Dict[str, Union[np.ndarray, List[np.ndarray]]]]]):
        """Set buffer of experience replay.

        Args:
            buffer: Internel state of the replay buffer.
        """
        if self.training:
            if buffer is not None:
                self.replay_buffer.set(buffer)
