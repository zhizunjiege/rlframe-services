from datetime import datetime
from typing import Union, Dict, Optional, List

import numpy as np
import tensorflow as tf
# from .replay.simple_replay import SimpleReplay
# import tensorflow_probability as tfp
# import NetworkOptimizer.optimizer_pb2 as pb2
#
# from NetworkOptimizer.NeutralNetwork import NeutralNetwork
# from NetworkOptimizer import ACTIVATE_FUNC, INITIALIZER, OPTIMIZER
from .base import RLModelBase
from .replay.simple_replay import SimpleReplay


class NormalNoise:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(scale=self.sigma, size=self.mu.shape)

    def reset(self):
        pass


class IPPO(RLModelBase):

    def __init__(
        self,
        training: bool,
        # networks: Dict[str, tf.keras.Model],
        *,
        obs_dim: int = 4,
        act_num: int = 2,
        hidden_layers: List[int] = [64, 64],
        lr: float = 0.001,
        gamma: float = 0.9,
        replay_size: int = 1000000,
        batch_size: int = 32,
        start_steps: int = 0,
        update_after: int = 32,
        update_online_every: int = 1,
        update_target_every: int = 200,
        seed: Optional[int] = None,
        agent_num: int = 4,
        noise_range: float = 0.1,
        action_bound: float = 0.5,
        epsilon: float = 0.2,
    ):
        super().__init__(training)

        self.lr = lr
        self.gamma = gamma
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_online_every = update_online_every
        self.update_target_every = update_target_every
        self.gamma = gamma
        self.seed = seed
        self._react_steps = 0
        self._train_steps = 0

        self.agent_num = agent_num
        self.noise_range = noise_range
        self.action_bound = action_bound
        self.epsilon = epsilon
        self.__nobs = obs_dim
        self.__nact = act_num
        self.__actor_list = []
        self.__actor_old_list = []
        self.__critic_list = []
        self.__actor_optimizer_list = []
        self.__critic_optimizer_list = []
        self.replay_buffer_list = []
        self.cumulative_reward_buffer = {}
        if training:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)

            for agent_index in range(self.agent_num):
                self.__actor_list.append(self.actor_net_builder('actor', obs_dim, hidden_layers, act_num))
                self.__actor_old_list.append(self.actor_net_builder('actor_old', obs_dim, hidden_layers, act_num))
                self.__actor_optimizer_list.append(tf.keras.optimizers.Adam(lr))
                self.__critic_list.append(self.critic_net_builder('critic', obs_dim, hidden_layers, 1))
                self.__critic_optimizer = tf.keras.optimizers.Adam(lr)
                self.cumulative_reward_buffer[agent_index] = []
            for agent_index in range(self.agent_num):
                self.replay_buffer_list.append(
                    SimpleReplay(self.__nobs, self.__nact, self.replay_size, dtype=np.float32))

            log_dir = f'logs/gradient_tape/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            self.summary_writer = tf.summary.create_file_writer(log_dir)
            tf.summary.trace_on(graph=True, profiler=True)
        else:
            for agent_index in range(self.agent_num):
                self.__actor_list.append(self.actor_net_builder('actor', obs_dim, hidden_layers, act_num))
                self.__actor_old_list = \
                    self.__critic_list = None
                self.replay_buffer_list = None

    def react(self, states: Dict[int, np.ndarray]):
        action_n = {}
        if self.training:
            for agent_index in range(self.agent_num):
                s = states[agent_index][np.newaxis, :]
                mu, sigma = self.__actor_list[agent_index](s, training=False)
                # u = tfp.distributions.Normal(logits[0], self.noise_range)
                # action = u.sample(1) #
                # action_n[agent_index] = np.squeeze(action, axis=0)
                # a = tf.clip_by_value(a, -self.action_bound + self.action_shift, self.action_bound + self.action_shift)
                # pi = tfp.distributions.Normal(mu, sigma)  # 用mu和sigma构建正态分布
                pi = tf.compat.v1.distributions.Normal(mu, sigma)
                a = tf.squeeze(pi.sample(1), axis=0)[0]
                action_n[agent_index] = np.clip(a, -self.action_bound, self.action_bound)

            self._react_steps += 1
        else:
            for agent_index in range(self.agent_num):
                s = states[agent_index][np.newaxis, :]
                logits = self.__actor_list[agent_index](s, training=False)
                action = logits[0]
                action_n[agent_index] = np.squeeze(action, axis=0)
            self._react_steps += 1
        return action_n

    def store(
        self,
        states: Dict[int, np.ndarray],
        # states: np.ndarray,
        actions,  #
        next_states: Dict[int, np.ndarray],
        reward: Dict[int, float],
        terminated: bool,
        truncated: bool,
    ):
        for agent_index in range(self.agent_num):
            self.replay_buffer_list[agent_index].store(states[agent_index], actions[agent_index], reward[agent_index],
                                                       next_states[agent_index], terminated)

    def __del__(self):
        """Close model."""
        ...

    def train(self):
        if self.replay_buffer_list[0].size >= self.update_after and self._react_steps % self.update_online_every == 0:
            for _ in range(self.update_online_every):
                exp_n = []
                adv = []
                for agent_index in range(self.agent_num):
                    exp = self.replay_buffer_list[agent_index].sample(self.batch_size)
                    exp_n.append(exp)
                    # batch = self.agent_list[0].replay_buffer.sample(self.batch_size)
                    v_s_ = self.__critic_list[agent_index](exp_n[agent_index]['obs2'])
                    self.cal_v_s_(exp_n, v_s_, agent_index)
                    cumulative_reward = self.cumulative_reward_buffer[agent_index]
                    r = np.array(cumulative_reward, np.float32)
                    # print(r.shape)
                    adv.append(self.cal_adv(exp_n, agent_index, r))
                self.__apply_gradients(exp_n, adv)
                # self.__apply_gradients()
                self._train_steps += 1
                # tf.Graph().finalize()
                # with self.summary_writer.as_default():
                #     with tf.summary.record_if(self._train_steps % 20 == 0):
                #         tf.summary.scalar('actor_loss', actor_loss, step=self._train_steps)
                #         tf.summary.scalar('critic_loss', critic_loss, step=self._train_steps)
                #         tf.summary.scalar('td_error', tf.reduce_mean(tf.abs(td_errors)), step=self._train_steps)

    def update_old_pi(self):
        """
        更新actor_old参数。
        """
        for agent_index in range(self.agent_num):
            for pi, oldpi in zip(self.__actor_list[agent_index].trainable_weights,
                                 self.__actor_old_list[agent_index].trainable_weights):
                oldpi.assign(pi)

    def get_weights(self):
        weights = {
            'actor': [self.__actor_list[agent_index].get_weights() for agent_index in range(self.agent_num)],
        }
        if self.training:
            weights['actor_old'] = [self.__actor_old_list[agent_index].get_weights() for agent_index in
                                    range(self.agent_num)]
        return weights

    def set_weights(self, weights):
        for agent_index in range(self.agent_num):
            self.__actor_list[agent_index].set_weights(weights['actor'][agent_index])
        if self.training and 'actor_old' in weights:
            for agent_index in range(self.agent_num):
                self.__actor_old_list[agent_index].set_weights(weights['actor_old'][agent_index])

    def actor_net_builder(self, name, input_dim, hidden_layers, output_dim):
        inputs = tf.keras.Input(shape=(input_dim,))
        outputs = inputs
        for layer in hidden_layers:
            outputs = tf.keras.layers.Dense(units=layer, activation='relu')(outputs)
        sigma = tf.keras.layers.Dense(units=output_dim, activation='softplus')(outputs)
        outputs = tf.keras.layers.Dense(units=output_dim, activation='tanh')(outputs)
        mu = tf.keras.layers.Lambda(lambda x: x * self.action_bound)(outputs)
        return tf.keras.Model(inputs=inputs, outputs=[mu, sigma], name=name)

    def critic_net_builder(self, name, input_dim, hidden_layers, output_dim):
        inputs = tf.keras.Input(shape=(input_dim,))
        outputs = inputs
        for layer in hidden_layers:
            outputs = tf.keras.layers.Dense(units=layer, activation='relu')(outputs)
        outputs = tf.keras.layers.Dense(units=output_dim, activation='linear')(outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    @tf.function()
    def __apply_gradients(self, exp_n, adv):
        for agent_index in range(self.agent_num):
            state = exp_n[agent_index]['obs1']
            # next_state = exp_n[agent_index]['obs2']
            action = exp_n[agent_index]['acts']

            # v_s_ = self.__critic_list[agent_index](next_state)
            # self.cal_vs_(exp_n, v_s_, agent_index)

            cumulative_reward = self.cumulative_reward_buffer[agent_index]
            # s = np.array(state, np.float32)
            # a = np.array(action, np.float32)
            r = np.array(cumulative_reward, np.float32)

            self.update_old_pi()
            # adv = (r - self.__critic_list[agent_index](state))
            # adv = self.cal_adv(adv)
            # adv = np.array(adv, np.float32)
            with tf.GradientTape() as tape:
                mu, sigma = self.__actor_list[agent_index](state)
                # pi = tfp.distributions.Normal(mu, sigma)
                pi = tf.compat.v1.distributions.Normal(mu, sigma)
                mu_old, sigma_old = self.__actor_old_list[agent_index](state)
                # oldpi = tfp.distributions.Normal(mu_old, sigma_old)
                oldpi = tf.compat.v1.distributions.Normal(mu_old, sigma_old)
                # ratio = pi.prob(action) / (oldpi.prob(action) + 1e-8)
                ratio = tf.exp(pi.prob(action) - oldpi.prob(action))
                # print(ratio.shape, adv[agent_index].shape)
                surr = ratio * adv[agent_index]
                # tf.print(surr.shape)
                actor_loss = -tf.reduce_mean(
                    tf.minimum(surr, tf.clip_by_value(
                        ratio, 1. - self.epsilon, 1. + self.epsilon) * adv[agent_index])
                )
            actor_grad = tape.gradient(actor_loss, self.__actor_list[agent_index].trainable_weights)
            self.__actor_optimizer_list[agent_index].apply_gradients(
                zip(actor_grad, self.__actor_list[agent_index].trainable_weights))

            with tf.GradientTape() as tape:
                advantage = r - self.__critic_list[agent_index](state)  # td-error
                loss = tf.reduce_mean(tf.square(advantage))
            grad = tape.gradient(loss, self.__critic_list[agent_index].trainable_weights)
            self.__critic_optimizer.apply_gradients(zip(grad, self.__critic_list[agent_index].trainable_weights))

            # v_s_ = self.__critic_list[agent_index](np.array([next_state], dtype=np.float32))[0, 0]
            # discounted_r = []
            # for rew in exp_n[agent_index]['rews'][::-1]:
            #     v_s_ = rew + self.gamma * v_s_
            #     discounted_r.append(v_s_)
            # discounted_r.reverse()
            # discounted_r = np.array(discounted_r)[:, np.newaxis]
            # self.cumulative_reward_buffer[agent_index].extend(discounted_r)

    # return actor_loss, critic_loss, td_errors

    def cal_adv(self, exp_n, agent_index, r):
        adv = (r - self.__critic_list[agent_index](exp_n[agent_index]['obs1']))
        # print(r.shape)
        adv = adv.numpy()
        # print(adv.shape)
        adv = np.array(adv, np.float32)
        return adv

    def cal_v_s_(self, exp_n, v_s_, agent_index):
        discounted_r = []
        self.cumulative_reward_buffer[agent_index] = []
        for rew in exp_n[agent_index]['rews'][::-1]:
            v_s_ = rew + self.gamma * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        # print(discounted_r[0].shape)
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        # print(discounted_r.shape)
        self.cumulative_reward_buffer[agent_index] = discounted_r

    def get_buffer(self) -> Dict[str, Union[int, Dict[str, np.ndarray]]]:
        """Get buffer of experience replay.

        Returns:
            Internel state of the simple replay buffer.
        """
        return self.replay_buffer_list[0].get()

    def set_buffer(self, buffer: Dict[str, Union[int, Dict[str, np.ndarray]]]) -> None:
        """Set buffer of experience replay.

        Args:
            buffer: Internel state of the simple replay buffer.
        """
        for agent_index in range(self.agent_num):
            self.replay_buffer_list[agent_index].set(buffer)
        self.replay_size = buffer['max_size']
