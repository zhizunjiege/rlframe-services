from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from models.base import RLModelBase
from models.replay.qmix_replay import QmixReplay
from models.qmix.qagent import QAgent
from models.qmix.policy import EpsGreedyQPolicy
from models.qmix.MixingNet import MixingNet


class QMIX(RLModelBase):
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
        agent_num: int = 2,
        tau: float = 0.01,
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
            update_target_every: Number of env interactions that should elapse between target network updates.
            seed: Seed for random number generators.
            agent_num: Numbers of agents
            tau: Parameter for soft update
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
        self.tau = tau
        self.agent_num = agent_num
        self.agent_list = []
        self.prev_state = None
        self.prev_observations = None
        self.last_q_values = [0]  # @todo
        self.last_targets = [0]  # @todo
        self.trainable_variables = None
        self.target_trainable_variables = None
        self.prev_obs1_list = []
        self.prev_obs2_list = []
        if training:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            trajectory_len = 1
            agent_input_shape = (trajectory_len, self.obs_dim)
            for agent_index in range(self.agent_num):
                model, self.eval_hidden = self.build_q_network(agent_input_shape, self.act_num, self.hidden_layers, 'online')
                target_model, self.target_eval_hidden = self.build_q_network(agent_input_shape, self.act_num, self.hidden_layers, 'target')
                agent = QAgent(
                    aid=agent_index,
                    policy=EpsGreedyQPolicy(eps=self.epsilon_max, eps_decay_rate=self.epsilon_decay,
                                            min_eps=self.epsilon_min),
                    model=model,
                    target_model=target_model,
                    eval_hidden=self.eval_hidden
                )
                agent.target_model.set_weights(model.get_weights())
                self.init_state = np.random.random((self.obs_dim,))
                self.agent_list.append(agent)
            # self.online_net = self.net_builder('online', obs_dim, hidden_layers, act_num)
            # self.target_net = self.net_builder('target', obs_dim, hidden_layers, act_num)
            # self.update_target()
            models = []
            target_models = []
            for agent in self.agent_list:
                models.append(agent.model)
                target_models.append(agent.target_model)
                if self.trainable_variables is None:
                    self.trainable_variables = agent.model.trainable_variables
                    self.target_trainable_variables = agent.target_model.trainable_variables
                else:
                    self.trainable_variables += agent.model.trainable_variables
                    self.target_trainable_variables += agent.target_model.trainable_variables

            self.Qtot_model = MixingNet(models, embed_shape=64)
            self.Qtot_target_model = MixingNet(target_models, embed_shape=64)
            self.trainable_variables += self.Qtot_model.trainable_variables
            self.target_trainable_variables += self.Qtot_target_model.trainable_variables
            # self._epsilon = epsilon_max
            self._react_steps = 0
            self._train_steps = 0

            # self.optimizer = tf.keras.optimizers.Adam(lr)
            self.replay_buffer = QmixReplay(obs_dim, act_num, replay_size, self.agent_num, dtype=np.float32)
            self.loss_fn = tf.keras.losses.MeanSquaredError()
            self.optimizer = tf.keras.optimizers.RMSprop()

            # self.log_dir = f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            # self.summary_writer = tf.summary.create_file_writer(self.log_dir)
            # self._graph_exported = False
            # tf.summary.trace_on(graph=True, profiler=False)
        else:
            self.online_net = self.net_builder('online', obs_dim, hidden_layers, act_num)

    def __del__(self):
        """Close DQN model."""
        ...

    def build_q_network(self, input_shape, nb_output, hidden_layers, name):
        input_layer = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Dense(hidden_layers[0], activation='relu')(input_layer)
        h = tf.keras.layers.GRU(hidden_layers[0], activation='relu')(x)
        # x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        output = tf.keras.layers.Dense(nb_output, activation='linear')(x)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output, name=name)

        return model, h

    def react(self, states: np.ndarray) -> list:
        """Get action.

        Args:
            states: States of enviroment.

        Returns:
            Action.
        """
        actions = []
        if self.training:
            for agent in self.agent_list:
                agent.reset(states)
            for agent in self.agent_list:
                action = agent.act()
                actions.append(action)
                # actions.append(np.random.random((8,)))
        self._react_steps += 1
        return actions

    def store(
        self,
        states: np.ndarray,
        actions,
        next_states: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Store experience repplay data.

        Args:
            states: States of enviroment.
            actions: Actions of model.
            next_states: Next states of enviroment.
            reward: Reward.
            terminated: Whether a `terminal state` (as defined under the MDP of the task) is reached.
            truncated: Whether a truncation condition outside the scope of the MDP is satisfied.
        """
        self.replay_buffer.store(states, actions, reward, next_states, terminated)

    def train(self):
        """Train model."""

        if self.replay_buffer.size >= self.update_after and self._react_steps % self.update_online_every == 0:
            for _ in range(self.update_online_every):

                idxs = np.random.randint(0, self.replay_buffer.size, size=self.batch_size)
                exp_n = self.replay_buffer.sample(idxs)
                states, next_states, actions, rewards, terminals = exp_n['obs1'], exp_n['obs2'], exp_n['acts'], exp_n[
                    'rews'], exp_n['term']
                observations = [[] for _ in range(self.agent_num)]
                next_observations = [[] for _ in range(self.agent_num)]

                for idx in idxs:
                    for i in range(self.agent_num):
                        observations[i].append(self.prev_obs1_list[idx][i])
                        next_observations[i].append(self.prev_obs2_list[idx][i])

                rewards = np.array(rewards).reshape(-1, 1)
                terminals = np.array(terminals).reshape(-1, 1)
                next_observations = np.array(next_observations)
                next_states = np.array(next_states)

                masks, target_masks = [], []
                for idxx, (agent, next_observation) in enumerate(zip(self.agent_list, next_observations)):
                    agent_out = agent.target_model(next_observation)
                    argmax_actions = tf.keras.backend.argmax(agent_out)
                    target_mask = tf.one_hot(
                        argmax_actions, depth=self.act_num)
                    target_masks.append(target_mask)
                    masks.append(actions[:, idxx, :])

                masks = tf.convert_to_tensor(masks)
                target_masks = tf.convert_to_tensor(target_masks)

                target_q_values = self.predict_on_batch(
                    next_states, next_observations, target_masks, self.Qtot_target_model)
                discounted_reward_batch = self.gamma * target_q_values * terminals
                targets = rewards + discounted_reward_batch

                observations = np.array(observations)
                states = np.array(states)
                states = tf.convert_to_tensor(states, dtype=tf.float32)
                observations = tf.convert_to_tensor(
                    observations, dtype=tf.float32)

                loss = self.train_on_batch(
                    states, observations, masks, targets)
                self.soft_update_target_model()

                self._train_steps += 1
                return loss
        # with self.summary_writer.as_default():
        #     tf.summary.scalar('loss', loss.numpy(), step=self._train_steps)
        #     # tf.summary.scalar('td_error', tf.math.reduce_mean(tf.math.abs(td_errors)), step=self._train_steps)
        #     if not self._graph_exported:
        #         tf.summary.trace_export(name='model', step=self._train_steps, profiler_outdir=self.log_dir)
        #         self._graph_exported = True

    def net_builder(self, name, input_dim, hidden_layers, output_dim):
        inputs = tf.keras.Input(shape=(input_dim,))
        outputs = inputs
        for layer in hidden_layers:
            outputs = tf.keras.layers.Dense(units=layer, activation='relu')(outputs)
        outputs = tf.keras.layers.Dense(units=output_dim, activation='linear')(outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    def predict_on_batch(
        self,
        states,
        observations,
        masks,
        model):
        q_values = model([observations, states, masks])
        return q_values

    def compute_q_values(self, state):
        q_values = self.Qtot_target_model.predict(np.array([state]))
        return q_values[0]

    @tf.function
    def train_on_batch(self, states, observations, masks, targets):
        with tf.GradientTape() as tape:
            tape.watch(observations)
            tape.watch(states)
            y_preds = self.Qtot_model([observations, states, masks])
            loss_value = self.loss_fn(targets, y_preds)

        self.last_q_values = y_preds  # @todo
        self.last_targets = targets  # @todo
        grads = tape.gradient(loss_value, self.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables))

        return loss_value
        # return loss_value.numpy()

    def soft_update_target_model(self):
        target_model_weights = np.array(self.Qtot_target_model.get_weights())
        model_weights = np.array(self.Qtot_model.get_weights())
        new_weight = (1. - self.tau) * target_model_weights \
                     + self.tau * model_weights
        self.Qtot_target_model.set_weights(new_weight)
        for agent in self.agent_list:
            agent.soft_update_target_model()

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get weights of neural networks.

        Returns:
            Weights of `online network` and `target network`(if exists).
        """
        weights = {
            'online': self.agent_list[0].model.get_weights(),
        }
        if self.training and self.agent_list[0].target_model is not None:
            weights['target'] = self.agent_list[0].target_model.get_weights()
        return weights

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set weights of neural networks.

        Args:
            weights: Weights of `online network` and `target network`(if exists).
        """
        self.agent_list[0].model.set_weights(weights['online'])
        if self.training and 'target' in weights:
            self.agent_list[0].target_model.set_weights(weights['target'])

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
