from datetime import datetime
from typing import Union, Dict, Optional, List

import numpy as np
import tensorflow as tf
# from .replay.simple_replay import SimpleReplay
# import tensorflow_probability as tfp
from models.base import RLModelBase
from models.replay.complex_replay import ComplexReplay


class NormalNoise:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(scale=self.sigma, size=self.mu.shape)

    def reset(self):
        pass


class MADDPG(RLModelBase):

    def __init__(
        self,
        training: bool,
        *,
        obs_dim: int = 4,
        act_num: int = 2,
        hidden_layers: List[int] = [64, 64],
        actor_lr: float = 0.001,
        critic_lr: float = 0.001,
        gamma: float = 0.9,
        replay_size: int = 1000000,
        batch_size: int = 32,
        start_steps: int = 0,
        update_after: int = 32,
        update_online_every: int = 1,
        update_target_every: int = 200,
        seed: Optional[int] = None,
        tau=0.001,
        agent_num: int = 4,
        noise_range: float = 0.1,
        action_span: float = 0.5
    ):
        """Init a MADDPG model.
        Args:
            training: Whether the model is in training mode.
            obs_dim: Dimension of observation.
            act_num: Number of actions.
            hidden_layers: Units of hidden layers.
            actor_lr: Actor's learning rate.
            critic_lr: Critic's learning rate.
            gamma: Discount factor.
            replay_size: Maximum size of replay buffer.
            batch_size: Size of batch.
            start_steps: Number of steps for uniform-random action selection before running real policy.
            update_after: Number of env interactions to collect before starting to do gradient descent updates.
            update_online_every: Number of env interactions that should elapse between gradient descent updates.
            update_target_every: Number of env interactions that should elapse between target network updates.
            seed: Seed for random number generators.
            tau: Parameter for soft-update
            agent_num: numbers of agents
            noise_range: Noise range that agents act in
            action_span: Allowed range for agent's action
        """
        super().__init__(training)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_online_every = update_online_every
        self.update_target_every = update_target_every
        self.gamma = gamma
        self.seed = seed
        self.tau = tau
        self._react_steps = 0
        self._train_steps = 0

        self.agent_num = agent_num
        self.noise_range = noise_range
        self.__nobs = obs_dim
        self.__nact = act_num
        self.__actor_list = []
        self.__actor_target_list = []
        self.__critic_list = []
        self.__critic_target_list = []
        self.__actor_optimizer_list = []
        self.__critic_optimizer_list = []
        self.replay_buffer_list = []
        self.action_span = action_span
        if training:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)

            for agent_index in range(self.agent_num):
                # self.__actor_list.append(networks['actor'])
                # self.__actor_target_list.append(networks['actor_target'])
                self.__actor_list.append(self.actor_net_builder('actor', obs_dim, hidden_layers, act_num,trainable=True))
                self.__actor_target_list.append(self.actor_net_builder('actor_target', obs_dim, hidden_layers, act_num, trainable=False))
                self.__actor_optimizer_list.append(tf.keras.optimizers.Adam(self.actor_lr))

                self.__update_target_weights(self.__actor_list[agent_index], self.__actor_target_list[agent_index],
                                             self.tau)

                # self.__critic_list.append(networks['critic'])
                # self.__critic_target_list.append(networks['critic_target'])
                self.__critic_list.append(
                    self.critic_net_builder('critic', self.agent_num * (obs_dim + act_num), hidden_layers, 1, trainable=True))
                self.__critic_target_list.append(
                    self.critic_net_builder('critic_target', self.agent_num * (obs_dim + act_num), hidden_layers, 1, trainable=False))
                self.__critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
                self.__update_target_weights(self.__critic_list[agent_index], self.__critic_target_list[agent_index],
                                             self.tau)

            for agent_index in range(self.agent_num):
                self.replay_buffer_list.append(
                    ComplexReplay(self.__nobs, self.__nact, self.replay_size, dtype=np.float32))

            # log_dir = f'logs/gradient_tape/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            # self.summary_writer = tf.summary.create_file_writer(log_dir)
            # self.writer = tf.summary.create_file_writer('board/maddpg_logs')
            # tf.summary.trace_on(graph=True, profiler=True)

        else:
            for agent_index in range(self.agent_num):
                self.__actor_list.append(self.actor_net_builder('actor', obs_dim, hidden_layers, act_num, trainable=True))
                self.__actor_target_list = \
                    self.__critic_target_list = self.__critic_list = None
                self.replay_buffer_list = None

    def react(self, states: Dict[int, np.ndarray]):
        action_n = {}
        if self.training:
            for agent_index in range(self.agent_num):
                s = states[agent_index][np.newaxis, :]
                logits = self.__actor_list[agent_index](s)
                # u = tfp.distributions.Normal(logits[0], self.noise_range)
                # action = u.sample(1) #
                # action_n[agent_index] = np.squeeze(action, axis=0)
                # a = tf.clip_by_value(a, -self.action_bound + self.action_shift, self.action_bound + self.action_shift)
                noise = NormalNoise(logits[0], self.noise_range)
                action = logits[0] + noise()
                action = tf.clip_by_value(action, clip_value_min=-self.action_span, clip_value_max=self.action_span)
                action_n[agent_index] = action

            self._react_steps += 1
        else:
            for agent_index in range(self.agent_num):
                s = states[agent_index][np.newaxis, :]
                logits = self.__actor_list[agent_index](s)
                action = logits[0]
                action_n[agent_index] = np.squeeze(action, axis=0)
            self._react_steps += 1
        return action_n

    def store(
        self,
        states: Dict[int, np.ndarray],
        actions,
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
                idxs = np.random.randint(0, self.replay_buffer_list[0].size, size=self.batch_size)
                for agent_index in range(self.agent_num):
                    exp = self.replay_buffer_list[agent_index].sample(idxs)
                    exp_n.append(exp)
                # batch = self.agent_list[0].replay_buffer.sample(self.batch_size)
                actor_loss = self.__apply_gradients(exp_n)
                self._train_steps += 1
                # with self.writer.as_default():
                #     with tf.summary.record_if(self._train_steps % 20 == 0):
                #         tf.summary.scalar('actor_loss', actor_loss, step=self._train_steps)
                #         tf.summary.scalar('critic_loss', critic_loss, step=self._train_steps)
                #         tf.summary.scalar('td_error', tf.reduce_mean(tf.abs(td_errors)), step=self._train_steps)
                return actor_loss

    def get_weights(self):
        weights = {
            'actor': [self.__actor_list[agent_index].get_weights() for agent_index in range(self.agent_num)],
        }
        if self.training:
            weights['actor_target'] = [self.__actor_target_list[agent_index].get_weights() for agent_index in
                                       range(self.agent_num)]
        return weights

    def set_weights(self, weights):
        for agent_index in range(self.agent_num):
            self.__actor_list[agent_index].set_weights(weights['actor'][agent_index])
        if self.training and 'actor_target' in weights:
            for agent_index in range(self.agent_num):
                self.__actor_target_list[agent_index].set_weights(weights['actor_target'][agent_index])

    def actor_net_builder(self, name, input_dim, hidden_layers, output_dim, trainable):
        inputs = tf.keras.Input(shape=(input_dim,))
        outputs = inputs
        for layer in hidden_layers:
            outputs = tf.keras.layers.Dense(units=layer, activation='relu')(outputs)
        outputs = tf.keras.layers.Dense(units=output_dim, activation='tanh')(outputs)
        actor_output = tf.keras.layers.Lambda(lambda x: x * np.array(self.action_span))(outputs)
        actor_model = tf.keras.Model(inputs=inputs, outputs=actor_output, name=name, trainable=trainable)
        actor_model.compile(optimizer=tf.keras.optimizers.Adam(self.actor_lr))
        # return tf.keras.Model(inputs=inputs, outputs=actor_output, name=name, trainable=trainable)
        return actor_model

    def critic_net_builder(self, name, input_dim, hidden_layers, output_dim, trainable):
        # inputs = tf.keras.Input(shape=(input_dim,))
        input_magent_s = tf.keras.Input(shape=(self.__nobs * self.agent_num,), dtype="float32")
        input_magent_a = tf.keras.Input(shape=(self.__nact * self.agent_num,), dtype="float32")
        input_critic = tf.concat([input_magent_s, input_magent_a], axis=-1)
        outputs = input_critic
        for layer in hidden_layers:
            outputs = tf.keras.layers.Dense(units=layer, activation='relu')(outputs)
        outputs = tf.keras.layers.Dense(units=output_dim)(outputs)
        critic_model = tf.keras.Model(inputs=[input_magent_s, input_magent_a], outputs=outputs, name=name, trainable=trainable)
        critic_model.compile(optimizer=tf.keras.optimizers.Adam(self.critic_lr))
        # return tf.keras.Model(inputs=[input_magent_s, input_magent_a], outputs=outputs, name=name, trainable=trainable)
        return critic_model

    def __update_target_weights(self, model, target_model, tau):
        weights = model.weights
        target_weights = target_model.weights
        [a.assign(a * (1 - tau) + b * tau) for a, b in zip(target_weights, weights)]


    @tf.function
    def __apply_gradients(self, exp_n):
        with tf.GradientTape(persistent=True) as tape:
            # action_list = []
            for agent_index in range(self.agent_num):
                obs = exp_n[agent_index]['obs1']
                # action = exp_n[agent_index]['acts']
                action = self.__actor_list[agent_index](obs)
                # action_list.append(action)
                if agent_index == 0:
                    all_obs = tf.convert_to_tensor(obs)
                    all_action = action
                else:
                    all_obs = tf.concat([all_obs, obs], axis=-1)
                    all_action = tf.concat([all_action, action], axis=-1)

            for agent_index in range(self.agent_num):
                obs_ = exp_n[agent_index]['obs2']  # 得到当前的状态obs --> exp_n[agent_index][1]
                action_ = self.__actor_target_list[agent_index](obs_)  # 得到当前agent关于自己的obs的动作值
                if agent_index == 0:
                    all_obs_ = tf.convert_to_tensor(obs_)
                    all_action_ = action_
                else:
                    all_obs_ = tf.concat([all_obs_, obs_], axis=-1)
                    all_action_ = tf.concat([all_action_, action_], axis=-1)

            # action = [[] for _ in range(self.batch_size)]
            # for idx in range(self.batch_size):
            #     for index in range(self.agent_num):
            #         action[idx].append(exp_n[index]['acts'][idx])
            for agent_index in range(self.agent_num):
                act = exp_n[agent_index]['acts']
                if agent_index == 0:
                    action = act
                else:
                    action = tf.concat([action, act], axis=-1)

            for agent_index in range(self.agent_num):
                actor_pred = self.__actor_list[agent_index]
                critic_pred = self.__critic_list[agent_index]
                critic_target = self.__critic_target_list[agent_index]

                reward = exp_n[agent_index]['rews']
                terminated = exp_n[agent_index]['term']
                # 更新actor,每一个智能体的actor需要他本身的critic_pred，输入状态动作，然后最大化这个值
                q_pred = critic_pred([all_obs, all_action])
                actor_pred_loss = - tf.math.reduce_mean(q_pred)
                gradients = tape.gradient(actor_pred_loss, actor_pred.trainable_variables)
                # self.__actor_optimizer_list[agent_index].apply_gradients(zip(gradients, actor_pred.trainable_variables))
                actor_pred.optimizer.apply_gradients(zip(gradients, actor_pred.trainable_variables))
                # 更新critic网络
                q_pred_critic = critic_pred([all_obs, action])
                q_target_critic = reward + self.gamma * critic_target([all_obs_, all_action_])*(1-terminated)
                loss_critic = tf.keras.losses.mse(q_target_critic, q_pred_critic)
                loss_critic = tf.reduce_mean(loss_critic)
                critic_gradients = tape.gradient(loss_critic, critic_pred.trainable_variables)
                # self.__critic_optimizer.apply_gradients(zip(critic_gradients, critic_pred.trainable_variables))
                critic_pred.optimizer.apply_gradients(zip(critic_gradients, critic_pred.trainable_variables))

                # current_q = critic_pred(tf.concat((all_obs, all_action), axis=1))  #
                # target_q = reward + self.gamma * critic_target(tf.concat((all_obs_, all_action_), axis=1)) * (
                #         1 - terminated)
                # td_errors = tf.stop_gradient(target_q) - current_q
                # critic_loss = tf.reduce_mean(tf.math.square(td_errors))
                # critic_gradients = tape.gradient(critic_loss, critic_pred.trainable_variables)
                # self.__critic_optimizer.apply_gradients(zip(critic_gradients, critic_pred.trainable_variables))
                #
                # sample_action = actor_pred(obs)  #
                # for sample_index in range(self.agent_num):
                #     if sample_index == 0 and agent_index == 0:
                #         all_sample_action = sample_action
                #     elif sample_index == 0 and agent_index != 0:
                #         all_sample_action = action_list[0]
                #     elif sample_index == agent_index:
                #         all_sample_action = tf.concat([all_sample_action, sample_action], axis=-1)
                #     else:
                #         all_sample_action = tf.concat([all_sample_action, action_list[sample_index]], axis=-1)
                #
                # sample_q = critic_pred(tf.concat((all_obs, all_sample_action), axis=1))
                # actor_loss = - tf.reduce_mean(sample_q)
                # gradients = tape.gradient(actor_loss, actor_pred.trainable_variables)
                # self.__actor_optimizer_list[agent_index].apply_gradients(zip(gradients, actor_pred.trainable_variables))

        for agent_index in range(self.agent_num):
            self.__update_target_weights(self.__actor_list[agent_index], self.__actor_target_list[agent_index],
                                         self.tau)
            self.__update_target_weights(self.__critic_list[agent_index], self.__critic_target_list[agent_index],
                                         self.tau)
        return loss_critic

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

