from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from .base_net import MLP
from copy import deepcopy
from models.base import RLModelBase
from .memory import ReplayMemory, Experience
from .MixingNet import MixingNet


class QMIX(RLModelBase):
    """QMIX model."""

    def __init__(
        self,
        training: bool,
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
        update_target_every: int = 1,
        seed: Optional[int] = None,
        agent_num: int = 2
    ):
        """Init a QMIX model.
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
            seed: Seed for random number generators.
            agent_num: Numbers of agents
        """
        super().__init__(training)
        self.training = training
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
        self.agent_num = agent_num
        self.dtype = 'float32'

        if self.training:
            # 神经网络
            self.eval_mlp = []
            self.eval_mlp = [MLP(self.act_num) for _ in range(self.agent_num)]
            self.target_mlp = deepcopy(self.eval_mlp)

            # 把agentsQ值加起来的网络
            self.eval_qmix_net = MixingNet(agent_num=self.agent_num,
                                           qmix_hidden_dim=self.hidden_layers[0],
                                           obs_dim=self.obs_dim
                                           )
            self.target_qmix_net = deepcopy(self.eval_qmix_net)

            self.epsilon = epsilon_max
            self._react_steps = 0
            self._train_steps = 0

            self.replay_buffer = ReplayMemory(self.replay_size, self.agent_num, self.obs_dim, self.act_num, self.dtype)

            self.trainable_variables = list(self.eval_qmix_net.trainable_variables)
            for x in self.eval_mlp:
                self.trainable_variables += list(x.trainable_variables)

            self.optimizer = tf.keras.optimizers.Adam(self.trainable_variables,
                                                      lr=self.lr)
            self._graph_exported = False
            self.log_dir = f'data/logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)
            tf.summary.trace_on(graph=True, profiler=False)
        else:
            self.eval_mlp = [MLP(self.act_num) for _ in range(self.agent_num)]

    def __del__(self):
        """Close QMIX model."""
        ...

    def react(self, states: np.ndarray) -> list:
        """Get action.

        Args:
            states: States of enviroment.

        Returns:
            Action.
        """
        obs = tf.convert_to_tensor(np.stack(states), dtype=tf.float32)
        obs = obs.numpy()
        actions = tf.zeros((self.agent_num, 1), dtype=tf.float32)
        # obs1 = np.array([obs]*self.agent_num)

        for i in range(self.agent_num):
            sb = tf.stop_gradient(obs[i])
            sb = tf.expand_dims(sb, axis=0)
            # print(tf.shape(sb))
            q_value = tf.squeeze(self.eval_mlp[i](sb))

            if self.training:
                self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
                if np.random.uniform() < self.epsilon:
                        act = np.random.choice(self.act_num)  # action是一个整数
                        act = tf.convert_to_tensor([act])
                else:
                        act = tf.argmax(q_value)
            else:
                act = tf.argmax(q_value)
            actions = tf.py_function(self.my_function, [actions, i, act], Tout=act.dtype)
            # actions[i, :] = act
        # actions = tf.squeeze(actions)
        self._react_steps += 1
        return tf.stop_gradient(actions).numpy()

    def my_function(self, actions, i, act):
        actions = actions.numpy()
        actions[i,:] = act
        return actions

    def store(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        reward: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store experience replay data.

        Args:
            states: States of enviroment.
            actions: Actions of model.
            next_states: Next states of enviroment.
            reward: Reward.
            terminated: Whether a `terminal state` (as defined under the MDP of the task) is reached.
            truncated: Whether a truncation condition outside the scope of the MDP is satisfied.
        """
        self.replay_buffer.store(states, actions, next_states, reward, terminated)


    # @tf.function
    def train(self):
        """Train model."""
        if self.replay_buffer.size >= self.update_after and self._react_steps % self.update_online_every == 0:
            transitions = self.replay_buffer.sample(self.batch_size)
            batch = Experience(*zip(*transitions))

            q_evals = []
            q_targets = []

            for agent in range(self.agent_num):
                non_final_mask = tf.convert_to_tensor(list(map(lambda s: s is not None,
                                                batch.next_states)))
                # state_batch: batch_size x n_agents x dim_obs
                state_batch = tf.stack(batch.states)
                # action_batch: batch_size x n_agents x 1
                action_batch = tf.stack(batch.actions)
                # reward_batch: batch_size x n_agents x 1
                reward_batch = tf.stack(batch.rewards)
                non_final_next_states = tf.stack(
                    [s for s in batch.next_states if s is not None])

                # print(action_batch)
                state_i = state_batch[:, agent, :]
                # current_Q: batch_size x action_dim
                current_Q = self.eval_mlp[agent](state_i)
                # print(current_Q, action_batch[:,agent,:])
                current_Q = tf.gather(current_Q, action_batch[:,agent,:], batch_dims=1) # TODO
                # print(current_Q)
                # print(current_Q, action_batch[:,agent,:])
                q_evals.append(current_Q)

                target_Q = tf.zeros(self.batch_size, dtype=tf.float32)
                next_state_i = non_final_next_states[:, agent, :]

                target_Q = tf.where(non_final_mask, tf.reduce_max(self.target_mlp[agent](next_state_i), axis=1), 0.0) # TODO
                # target_Q[non_final_mask] = tf.reduce_max(self.target_mlp[agent](next_state_i), axis=1)[0]
                target_Q = tf.reshape(target_Q, (self.batch_size, 1))
                q_targets.append(target_Q)
            q_evals = tf.stack(q_evals, axis=1)
            q_targets = tf.stack(q_targets, axis=1)

            loss_Q = self.apply_grads(q_evals, state_batch, q_targets, non_final_next_states, reward_batch)

            self._train_steps+=1

            if self._train_steps % self.update_target_every == 0:
                self.target_qmix_net.set_weights(self.eval_qmix_net.get_weights())
            for i in range(self.agent_num):
                self.target_mlp[i].set_weights(self.eval_mlp[i].get_weights())

            with self.summary_writer.as_default():
                tf.summary.scalar('loss', tf.reduce_mean(loss_Q), step=self._train_steps)
                # tf.summary.scalar('td_error', tf.math.reduce_mean(tf.math.abs(td_errors)), step=self._train_steps)
                if not self._graph_exported:
                    tf.summary.trace_export(name='QMIX model', step=self._train_steps, profiler_outdir=self.log_dir)
                    self._graph_exported = True

    @tf.function
    def apply_grads(self, q_evals, state_batch, q_targets, non_final_next_states, reward_batch):
        with tf.GradientTape() as tape:
            q_total_eval = self.eval_qmix_net(q_evals, state_batch, self.batch_size)
            q_total_target = self.target_qmix_net(q_targets, non_final_next_states, self.batch_size)
            # print(reward_batch)
            targets = tf.expand_dims(reward_batch, axis=1) + self.gamma * q_total_target

            loss_Q = tf.keras.losses.mean_squared_error(tf.stop_gradient(targets), q_total_eval)

        gradients = tape.gradient(loss_Q, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss_Q

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get weights of neural networks.

        Returns:
            Weights of `online network` and `target network`(if exists).
        """
        # for i in range(self.agent_num):
        #     print(np.array(self.eval_mlp[i].get_weights()).shape)

        weights = {
            'online': [self.eval_mlp[i].get_weights() for i in range(self.agent_num)],
            'qmix_online': self.eval_qmix_net.get_weights()
        }
        if self.training and self.target_qmix_net is not None:
            weights['target'] = [self.target_mlp[i].get_weights() for i in range(self.agent_num)]
            weights['qmix_target'] = self.target_qmix_net.get_weights()
        return weights

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set weights of neural networks.

        Args:
            weights: Weights of `online network` and `target network`(if exists).
        """
        self.eval_qmix_net.set_weights(weights['qmix_online'])
        if self.training and 'qmix_target' in weights:
            self.target_qmix_net.set_weights(weights['qmix_target'])

        for i in range(self.agent_num):
            self.eval_mlp[i].set_weights(weights['online'][i])
            if self.training and 'target' in weights:
                self.target_mlp[i].set_weights(weights['target'][i])

    def get_buffer(self) -> Dict[str, int | Dict[str, np.ndarray]]:
        """Get buffer of experience replay.

        Returns:
            Internel state of the simple replay buffer.
        """
        return self.replay_buffer.get()

    def set_buffer(self, buffer: Dict[str, int | Dict[str, np.ndarray]]) -> None:
        """Set buffer of experience replay.

        Args:
            buffer: Internel state of the simple replay buffer.
        """
        self.replay_buffer.set(buffer)
        self.replay_size = buffer['max_size']
