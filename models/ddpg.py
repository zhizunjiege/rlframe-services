from datetime import datetime
from typing import Union, Dict, Optional, List

import numpy as np
import tensorflow as tf
from .replay.simple_replay import SimpleReplay
import tensorflow_probability as tfp
# import NetworkOptimizer.optimizer_pb2 as pb2
#
# from NetworkOptimizer.NeutralNetwork import NeutralNetwork
# from NetworkOptimizer import ACTIVATE_FUNC, INITIALIZER, OPTIMIZER
from .base import RLModelBase


class NormalNoise:

    def __init__(self, mu, sigma=0.15):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(scale=self.sigma, size=self.mu.shape)

    def reset(self):
        pass


class DDPG(RLModelBase):

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
        tau=0.001,
        noise_range=0.1,
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
        self.tau = tau
        self._react_steps = 0
        self._train_steps = 0
        self.__nobs = obs_dim
        self.__nact = act_num
        self.noise_range = noise_range

        if training:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)

            # self.__react_steps = 0
            # self.__train_steps = 0

            self.__actor = self.actor_net_builder('actor', obs_dim, hidden_layers, act_num)
            self.__actor_target = self.actor_net_builder('actor_target', obs_dim, hidden_layers, act_num)
            self.__actor_optimizer = tf.keras.optimizers.Adam(lr)
            self.__update_target_weights(self.__actor, self.__actor_target, self.tau)

            self.__critic = self.critic_net_builder('critic', (obs_dim + act_num), hidden_layers, 1)
            self.__critic_target = self.critic_net_builder('critic_target', (obs_dim + act_num), hidden_layers, 1)
            self.__critic_optimizer = tf.keras.optimizers.Adam(lr)
            self.__update_target_weights(self.__critic, self.__critic_target, self.tau)

            # self.__nobs = self.__actor.layers[0].input_shape[0][1]
            # self.__nact = self.__actor.layers[-1].output_shape[1]
            # self.noise = NormalNoise(mu=np.zeros(self.__nact), sigma=0.15)
            # self.optimizer = tf.keras.optimizers.Adam(lr)
            self.replay_buffer = SimpleReplay(self.__nobs, self.__nact, self.replay_size, dtype=np.float32)

            log_dir = f'logs/gradient_tape/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            self.summary_writer = tf.summary.create_file_writer(log_dir)
            tf.summary.trace_on(graph=True, profiler=True)
        else:
            self.__actor = self.actor_net_builder('actor', obs_dim, hidden_layers, act_num)
            self.__actor_target = self.__critic_target = self.__critic = None
            self.replay_buffer = None

    def react(self, states: np.ndarray):
        if self.training:
            s = states[np.newaxis, :]
            logits = self.__actor(s, training=False)
            # noise = NormalNoise(logits[0], self.noise_range)
            # action = logits[0] + noise()

            u = tfp.distributions.Normal(logits[0], 0.1)
            # action = tf.squeeze(u.sample(1), axis=0)[0] #
            action = u.sample(1)  #
            # states = np.expand_dims(states, axis=0).astype(np.float32)
            # a = self.__actor.predict(states)
            # a += self.noise()
            # a = tf.clip_by_value(a, -self.action_bound + self.action_shift, self.action_bound + self.action_shift)
            # action = np.argmax(a[0])
            self._react_steps += 1
        else:
            states = states[np.newaxis, :]
            logits = self.__actor(states, training=False)
            # action = np.argmax(logits[0])
            action = logits[0]
            self._react_steps += 1
        return np.squeeze(action, axis=0)
        # return action

    def store(
        self,
        states: np.ndarray,
        actions,
        next_states: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ):
        self.replay_buffer.store(states, actions, reward, next_states, terminated)

    def __del__(self):
        """Close model."""
        ...

    def train(self):
        if self.replay_buffer.size >= self.update_after and self._react_steps % self.update_online_every == 0:
            for _ in range(self.update_online_every):
                batch = self.replay_buffer.sample(self.batch_size)
                actor_loss, critic_loss, td_errors = self.__apply_gradients(
                    batch['obs1'],
                    batch['acts'],
                    batch['obs2'],
                    batch['rews'],
                    batch['term'],
                )

                self._train_steps += 1

                with self.summary_writer.as_default():
                    with tf.summary.record_if(self._train_steps % 20 == 0):
                        tf.summary.scalar('actor_loss', actor_loss, step=self._train_steps)
                        tf.summary.scalar('critic_loss', critic_loss, step=self._train_steps)
                        tf.summary.scalar('td_error', tf.reduce_mean(tf.abs(td_errors)), step=self._train_steps)

    def get_weights(self):
        weights = {
            'actor': self.__actor.get_weights(),
        }
        if self.training is not None:
            weights['actor_target'] = self.__actor_target.get_weights()
        return weights

    def set_weights(self, weights):
        self.__actor.set_weights(weights['actor'])
        if self.training and 'actor_target' in weights:
            self.__actor_target.set_weights(weights['actor_target'])

    # def __build_net(self, name, network):
    #     input_layer = None
    #     hidden_layers = None
    #     output_layer = None
    #     for index, layer in enumerate(network.layers):
    #         layer_name = f'{name}_layer_{index}'
    #         if layer.type == pb2.Layer.LayerType.input:
    #             input_layer = tf.keras.layers.Input(
    #                 shape=(layer.neutralAmount,),
    #                 name=layer_name,
    #             )
    #             hidden_layers = input_layer
    #         elif layer.type == pb2.Layer.LayerType.affine:
    #             hidden_layers = tf.keras.layers.Dense(
    #                 units=layer.neutralAmount,
    #                 activation=ACTIVATE_FUNC[layer.activator],
    #                 kernel_initializer=INITIALIZER[network.initialization],
    #                 bias_initializer=INITIALIZER[network.initialization],
    #                 name=layer_name,
    #             )(hidden_layers)
    #             if layer.appendix.appendixType == pb2.Layer.Appendix.AppendixType.dropout:
    #                 hidden_layers = tf.keras.layers.Dropout(layer.appendix.param)(hidden_layers)
    #             elif layer.appendix.appendixType == pb2.Layer.Appendix.AppendixType.normalbatch:
    #                 hidden_layers = tf.keras.layers.BatchNormalization()(hidden_layers, training=True)
    #             elif layer.appendix.appendixType == pb2.Layer.Appendix.AppendixType.ic:
    #                 hidden_layers = tf.keras.layers.Dropout(layer.appendix.param)(hidden_layers)
    #                 hidden_layers = tf.keras.layers.BatchNormalization()(hidden_layers, training=True)
    #         elif layer.type == pb2.Layer.LayerType.output:
    #             output_layer = tf.keras.layers.Dense(
    #                 units=layer.neutralAmount,
    #                 activation=ACTIVATE_FUNC[layer.activator],
    #                 kernel_initializer=INITIALIZER[network.initialization],
    #                 bias_initializer=INITIALIZER[network.initialization],
    #                 name=layer_name,
    #             )(hidden_layers)
    #     return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
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

    def __update_target_weights(self, model, target_model, tau):
        weights = model.weights
        target_weights = target_model.weights
        [a.assign(a * (1 - tau) + b * tau) for a, b in zip(target_weights, weights)]

    @tf.function
    def __apply_gradients(self, obs, act, next_obs, rew, terminated):
        with tf.GradientTape() as tape:
            next_action_target = self.__actor_target(next_obs)
            next_q_target = self.__critic_target(tf.concat((next_obs, next_action_target), axis=1))
            target_q = rew + self.gamma * next_q_target * (1 - terminated)
            current_q = self.__critic(tf.concat((obs, act), axis=1))
            td_errors = tf.stop_gradient(target_q) - current_q
            critic_loss = tf.reduce_mean(tf.math.square(td_errors))
        critic_grad = tape.gradient(critic_loss, self.__critic.trainable_variables)
        self.__critic_optimizer.apply_gradients(zip(critic_grad, self.__critic.trainable_variables))

        with tf.GradientTape() as tape:
            sample_action = self.__actor(obs)
            actor_loss = -tf.reduce_mean(self.__critic(tf.concat((obs, sample_action), axis=1)))
        actor_grad = tape.gradient(actor_loss, self.__actor.trainable_variables)
        self.__actor_optimizer.apply_gradients(zip(actor_grad, self.__actor.trainable_variables))

        self.__update_target_weights(self.__actor, self.__actor_target, self.tau)
        self.__update_target_weights(self.__critic, self.__critic_target, self.tau)
        return actor_loss, critic_loss, td_errors

    def get_buffer(self) -> Dict[str, Union[int, Dict[str, np.ndarray]]]:
        """Get buffer of experience replay.

        Returns:
            Internel state of the simple replay buffer.
        """
        return self.replay_buffer.get()

    def set_buffer(self, buffer: Dict[str, Union[int, Dict[str, np.ndarray]]]) -> None:
        """Set buffer of experience replay.

        Args:
            buffer: Internel state of the simple replay buffer.
        """
        self.replay_buffer.set(buffer)
        self.replay_size = buffer['max_size']
