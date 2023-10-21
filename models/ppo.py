from typing import Dict, List, Literal, Optional, Tuple, Union

from keras import optimizers
import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd

from .base import RLModelBase
from .core import MLPModel, discount_cumsum


class MLPDiscretePolicy(MLPModel):

    def __init__(self, name: str, trainable: bool, obs_dim: int, hidden_layers: List[int], act_dim: int):
        super().__init__(name, trainable, obs_dim, hidden_layers, 'relu', act_dim, 'softmax')

    @tf.function
    def call(self, obs: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        logits = super().call(obs, training)
        if training:
            dist = tfd.Categorical(probs=logits)
            act = dist.sample()
            logp = dist.log_prob(act)
        else:
            act = tf.argmax(logits, axis=1, output_type=tf.int32)
            logp = None
        return act, logp  # (batch,), (batch,)

    @tf.function
    def logp(self, obs: tf.Tensor, act: tf.Tensor) -> tf.Tensor:
        logits = super().call(obs, True)
        dist = tfd.Categorical(probs=logits)
        logp = dist.log_prob(tf.squeeze(tf.cast(act, tf.int32), axis=1))
        return logp  # (batch,)


class MLPContinuousPolicy(MLPModel):

    def __init__(self, name: str, trainable: bool, obs_dim: int, hidden_layers: List[int], act_dim: int):
        super().__init__(name, trainable, obs_dim, hidden_layers, 'relu', act_dim, 'tanh')
        log_std = -0.5 * tf.ones(act_dim, dtype=tf.float32)
        self.log_std = tf.Variable(log_std, trainable=True)

    @tf.function
    def call(self, obs: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        mu = super().call(obs, training)
        if training:
            dist = tfd.Normal(mu, tf.exp(self.log_std))
            act = dist.sample()
            logp = tf.reduce_sum(dist.log_prob(act), axis=1)
        else:
            act = mu
            logp = None
        return act, logp  # (batch, act_dim), (batch,)

    @tf.function
    def logp(self, obs: tf.Tensor, act: tf.Tensor) -> tf.Tensor:
        mu = super().call(obs, True)
        dist = tfd.Normal(mu, tf.exp(self.log_std))
        logp = tf.reduce_sum(dist.log_prob(act), axis=1)
        return logp  # (batch, )


class MLPMultiDiscretePolicy(MLPModel):

    def __init__(self, name: str, trainable: bool, obs_dim: int, hidden_layers: List[int], act_dims: List[int]):
        super().__init__(name, trainable, obs_dim, hidden_layers, 'relu', act_dims, 'softmax')

    @tf.function
    def call(self, obs: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        multi_logits = super().call(obs, training)
        if training:
            dists = [tfd.Categorical(probs=logits) for logits in multi_logits]
            acts = [dist.sample() for dist in dists]
            logp = tf.reduce_sum([dist.log_prob(act) for dist, act in zip(dists, acts)], axis=0)
        else:
            acts = [tf.argmax(logits, axis=1, output_type=tf.int32) for logits in multi_logits]
            logp = None
        return tf.stack(acts, axis=1), logp  # (batch, len(act_dims)), (batch,)

    @tf.function
    def logp(self, obs: tf.Tensor, acts: tf.Tensor) -> tf.Tensor:
        multi_logits = super().call(obs, True)
        dists = [tfd.Categorical(probs=logits) for logits in multi_logits]
        acts = tf.split(tf.cast(acts, tf.int32), acts.shape[1], axis=1)
        logp = tf.reduce_sum([dist.log_prob(tf.squeeze(act, axis=1)) for dist, act in zip(dists, acts)], axis=0)
        return logp  # (batch, )


class MLPHybridPolicy(MLPModel):

    def __init__(self, name: str, trainable: bool, obs_dim: int, hidden_layers: List[int], act_dims: List[List[int]]):
        adims = [len(act_dims), len(act_dims[0])]
        super().__init__(name, trainable, obs_dim, hidden_layers, 'relu', adims, ['softmax', 'tanh'])
        multi_log_std = -0.5 * tf.ones(len(act_dims[0]), dtype=tf.float32)
        self.multi_log_std = tf.Variable(multi_log_std, trainable=True)

        self.masks = tf.constant(act_dims, dtype=tf.float32)

    @tf.function
    def call(self, obs: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        logits, multi_mu = super().call(obs, training)

        if training:
            dist_discrete = tfd.Categorical(probs=logits)
            act_discrete = dist_discrete.sample()
            logp_discrete = dist_discrete.log_prob(act_discrete)

            dist_continuous = tfd.Normal(multi_mu, tf.exp(self.multi_log_std))
            act_continuous = dist_continuous.sample()
            logp_continuous = dist_continuous.log_prob(act_continuous)

            masks = tf.gather(self.masks, act_discrete)
            logp = logp_discrete + tf.reduce_sum(logp_continuous * masks, axis=1)
        else:
            act_discrete = tf.argmax(logits, axis=1, output_type=tf.int32)
            act_continuous = multi_mu
            logp = None

        act_discrete = tf.cast(act_discrete[:, tf.newaxis], tf.float32)
        return tf.concat([act_discrete, act_continuous], axis=1), logp  # (batch, 1 + len(act_dims[0])), (batch,)

    @tf.function
    def logp(self, obs: tf.Tensor, acts: tf.Tensor) -> tf.Tensor:
        logits, multi_mu = super().call(obs, True)

        dist_discrete = tfd.Categorical(probs=logits)
        act_discrete = tf.cast(acts[:, 0], tf.int32)
        logp_discrete = dist_discrete.log_prob(act_discrete)

        dist_continuous = tfd.Normal(multi_mu, tf.exp(self.multi_log_std))
        act_continuous = tf.cast(acts[:, 1:], tf.float32)
        logp_continuous = dist_continuous.log_prob(act_continuous)

        masks = tf.gather(self.masks, act_discrete)
        logp = logp_discrete + tf.reduce_sum(logp_continuous * masks, axis=1)

        return logp  # (batch,)


PolicyType = Literal['discrete', 'continuous', 'multi-discrete', 'hybrid']

policies = {
    'discrete': MLPDiscretePolicy,
    'continuous': MLPContinuousPolicy,
    'multi-discrete': MLPMultiDiscretePolicy,
    'hybrid': MLPHybridPolicy,
}


class PPOBuffer:
    """A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.

    It inherited codes from spinningup project: https://github.com/openai/spinningup.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_size: int,
        gamma=0.99,
        lam=0.95,
        dtype=np.float32,
    ):
        """Init a ppo buffer.

        Args:
            obs_dim: Dimension of observation space.
            act_dim: Dimension of action space.
            max_size: Maximum size of buffer.
            gamma: Discount factor.
            lam: Lambda for Generalized Advantage Estimation.
            dtype: Data type of buffer.
        """
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=dtype)
        self.act_buf = np.zeros((max_size, act_dim), dtype=dtype)
        self.ret_buf = np.zeros(max_size, dtype=dtype)
        self.adv_buf = np.zeros(max_size, dtype=dtype)
        self.lgp_buf = np.zeros(max_size, dtype=dtype)
        self.max_size, self.gamma, self.lam = max_size, gamma, lam
        self.path_start_idx, self.ptr = 0, 0

    @property
    def size(self) -> int:
        """Size of buffer.

        Returns:
            Size of buffer.
        """
        return min(self.ptr, self.max_size)

    @property
    def full(self) -> bool:
        """Whether the buffer is full.

        Returns:
            Whether the buffer is full.
        """
        return self.ptr >= self.max_size

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        val: float,
        lgp: float,
    ):
        """Append one timestep of agent-environment interaction to the buffer.

        Args:
            obs: Observation.
            act: Action.
            rew: Reward.
            val: Value.
            lgp: Log probability.
        """
        ptr = self.ptr % self.max_size
        self.obs_buf[ptr] = obs
        self.act_buf[ptr] = act
        self.ret_buf[ptr] = rew  # shared with rew
        self.adv_buf[ptr] = val  # shared with val
        self.lgp_buf[ptr] = lgp
        self.ptr += 1

    def finish(self, last_val=0.0):
        """Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        Args:
            last_val: Estimation of value of the last state.
                Note: It should be 0 if the trajectory ended, otherwise V(s_T).
        """
        path_slice = np.arange(self.path_start_idx, self.ptr) % self.max_size
        rews = np.append(self.ret_buf[path_slice], last_val)
        vals = np.append(self.adv_buf[path_slice], last_val)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        self.path_start_idx = self.ptr

    def sample(self) -> Dict[str, np.ndarray]:
        """Sample all data from buffer.

        Returns:
            Sampled data.
        """
        assert self.full  # buffer has to be full before you can use
        self.path_start_idx, self.ptr = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            lgp=self.lgp_buf,
        )

    def get(self) -> Dict[str, Union[int, np.ndarray]]:
        """Get the internal state of the buffer.

        Returns:
            Internal state of the buffer.
        """
        return dict(
            path_start_idx=self.path_start_idx,
            ptr=self.ptr,
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            lgp=self.lgp_buf,
        )

    def set(self, state: Dict[str, Union[int, np.ndarray]]):
        """Set the internal state of the buffer.

        Args:
            state: Internal state of the buffer.
        """
        self.path_start_idx, self.ptr = state['path_start_idx'], state['ptr']
        self.obs_buf = state['obs']
        self.act_buf = state['act']
        self.ret_buf = state['ret']
        self.adv_buf = state['adv']
        self.lgp_buf = state['lgp']


class PPO(RLModelBase):
    """Proximal Policy Optimization model."""

    def __init__(
        self,
        training: bool,
        *,
        policy: PolicyType,
        obs_dim: int,
        act_dim: Union[int, List[int], List[List[int]]],
        hidden_layers_pi: List[int] = [64, 64],
        hidden_layers_vf: List[int] = [64, 64],
        lr_pi=0.0003,
        lr_vf=0.001,
        gamma=0.99,
        lam=0.97,
        epsilon=0.2,
        buffer_size=4000,
        update_pi_iter=80,
        update_vf_iter=80,
        max_kl=0.01,
        seed: Optional[int] = None,
    ):
        """Init a PPO model.

        Args:
            training: Whether the model is in training mode.

            policy: Type of Policy network.
                Note: one of `discrete`, `continuous`, `multi-discrete` and `hybrid`.
            obs_dim: Dimension of observation.
            act_dim: Dimension of actions.
                Note: it should be a list if policy is `multi-discrete` or a list of list if policy is `hybrid`.
            hidden_layers_pi: Units of hidden layers for policy network.
            hidden_layers_vf: Units of hidden layers for value network.
            lr_pi: Learning rate for policy network.
            lr_vf: Learning rate for value network.
            gamma: Discount factor.
            lam: Lambda for Generalized Advantage Estimation.
            epsilon: Clip ratio for PPO-cilp version.
            buffer_size: Size of buffer.
            update_pi_iter: Number of iterations for updating policy network.
            update_vf_iter: Number of iterations for updating value network.
            max_kl: Maximum value of kl divergence.
            seed: Seed for random number generators.
        """
        super().__init__(training)

        self.policy = policy
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers_pi = hidden_layers_pi
        self.hidden_layers_vf = hidden_layers_vf
        self.lr_pi = lr_pi
        self.lr_vf = lr_vf
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.update_pi_iter = update_pi_iter
        self.update_vf_iter = update_vf_iter
        self.max_kl = max_kl
        self.seed = seed

        if training:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)

            self.pi = policies[policy]('pi', True, obs_dim, hidden_layers_pi, act_dim)
            self.vf = MLPModel('vf', True, obs_dim, hidden_layers_vf, 'relu', 1, 'linear')
            self.pi_optimizer = optimizers.Adam(lr_pi)
            self.vf_optimizer = optimizers.Adam(lr_vf)

            if policy == 'discrete':
                self.adim = 1
            elif policy == 'continuous':
                self.adim = act_dim
            elif policy == 'multi-discrete':
                self.adim = len(act_dim)
            elif policy == 'hybrid':
                self.adim = 1 + len(act_dim[0])
            self.buffer = PPOBuffer(obs_dim, self.adim, buffer_size, gamma, lam)

            self.vals, self.lgps = [], []

            self._react_steps = 0
            self._train_pi_steps = 0
            self._train_vf_steps = 0
        else:
            self.pi = policies[policy]('pi', obs_dim, False, hidden_layers_pi, act_dim)

    def react(self, states: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get action.

        Args:
            states: States of enviroment.

        Returns:
            Action.
        """
        obs = states[np.newaxis, :]
        act, lgp = self.pi(obs, self.training)
        if self.training:
            val = self.vf(obs, self.training)
            self.vals.append(val[0])
            self.lgps.append(lgp[0])

            self._react_steps += 1
        return np.array(act[0])

    def store(
        self,
        states: np.ndarray,
        actions: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
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
        self.buffer.store(states, actions, reward, self.vals.pop(0), self.lgps.pop(0))

        if terminated or truncated or self.buffer.full:
            last_val = 0.0 if terminated else self.vf(next_states[np.newaxis, :], self.training)[0]
            self.buffer.finish(last_val)

    def train(self):
        """Train model.

        Returns:
            Losses of pi and vf.
        """
        losses = {}
        if self.buffer.full:
            data = self.buffer.sample()
            losses_pi, approx_kls = self.apply_pi_grads(data['obs'], data['act'], data['adv'], data['lgp'])
            self._train_pi_steps += losses_pi.shape[0]
            losses['loss_pi'] = np.array(losses_pi).tolist()
            losses['approx_kl'] = np.array(approx_kls).tolist()

            losses_vf = self.apply_vf_grads(data['obs'], data['ret'])
            self._train_vf_steps += losses_vf.shape[0]
            losses['loss_vf'] = np.array(losses_vf).tolist()
        return losses

    @tf.function
    def apply_pi_grads(self, obs, act, adv, lgp):
        losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        approx_kls = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        for step in tf.range(self.update_pi_iter):
            with tf.GradientTape() as tape:
                lgp_new = self.pi.logp(obs, act)
                ratio = tf.exp(lgp_new - lgp)
                clip_adv = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
                loss = -tf.reduce_mean(tf.minimum(ratio * adv, clip_adv))
            approx_kl = tf.reduce_mean(lgp - lgp_new)
            if approx_kl > 1.5 * self.max_kl:
                break
            grads = tape.gradient(loss, self.pi.trainable_variables)
            self.pi_optimizer.apply_gradients(zip(grads, self.pi.trainable_variables))
            losses = losses.write(step, loss)
            approx_kls = approx_kls.write(step, approx_kl)

        return losses.stack(), approx_kls.stack()

    @tf.function
    def apply_vf_grads(self, obs, ret):
        losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        for step in tf.range(self.update_vf_iter):
            with tf.GradientTape() as tape:
                val = tf.squeeze(self.vf(obs, True), axis=1)
                loss = tf.reduce_mean(tf.square(ret - val))
            grads = tape.gradient(loss, self.vf.trainable_variables)
            self.vf_optimizer.apply_gradients(zip(grads, self.vf.trainable_variables))
            losses = losses.write(step, loss)

        return losses.stack()

    def get_weights(self):
        """Get weights of neural networks.

        Returns:
            Weights of `pi` and `vf`(if exists).
        """
        weights = {
            'pi': self.pi.get_weights(),
        }
        if self.training:
            weights['vf'] = self.vf.get_weights()
        return weights

    def set_weights(self, weights):
        """Set weights of neural networks.

        Args:
            weights: Weights of `pi` and `vf`(if exists).
        """
        if 'pi' in weights:
            self.pi.set_weights(weights['pi'])
        if self.training:
            if 'vf' in weights:
                self.vf.set_weights(weights['vf'])

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
