import json
import time
import copy
import unittest
import tensorflow as tf
from typing import Union, Dict, Optional
from models.two_step_env import TwoStepEnv
import tqdm
import numpy as np
from matplotlib import pyplot as plt
from models.qmix.qmix import QMIX


class QMIXModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        with open('tests/models/qmix/hypers.json', 'r') as f:
            hypers = json.load(f)
        cls.model = QMIX(training=True, **hypers['hypers'])

    @classmethod
    def tearDownClass(cls):
        cls.model.close()
        cls.model = None

    def test_00_init(self):
        self.assertTrue(self.model.training)
        # self.assertEqual(self.model.__nobs, 12)
        # self.assertEqual(self.model.__nact, 8)

    def test_01_react(self):
        states = np.random.random((12,))
        t1 = time.time()
        for _ in range(1000):
            action_n = self.model.react(states)
        t2 = time.time()
        print(f'12x256x256x8 nn 1000 react time: {t2 - t1:.2f}s')
        # self.assertIsInstance(action_n[0], tf.Tensor)
        print(action_n[0])
        # self.assertEqual(action_n[0].shape, 8)

    def test_02_store(self):
        terminated = False
        truncated = False
        states = np.random.random((12,))
        action = [0, 0]
        next_states = np.random.random((12,))
        reward = 0.0
        for _ in range(1000):
            observations = []
            for agent in self.model.agent_list:
                agent.observe(states)
                observations.append(agent.trajectory)

            if self.model.prev_state is None:
                self.model.prev_state = states
                self.model.prev_observations = observations

            one_hot_actions = []
            for act in action:
                act = tf.one_hot(act, depth=self.model.act_num)
                one_hot_actions.append(act)

            self.model.store(self.model.prev_state, one_hot_actions, states, reward, terminated, truncated)
            self.model.prev_obs1_list.append(self.model.prev_observations)
            self.model.prev_obs2_list.append(observations)

            self.model.prev_state = states
            self.model.prev_observations = observations

        self.assertEqual(self.model.replay_buffer.size, 1000)

    def test_03_train(self):
        t1 = time.time()
        for _ in range(1000):
            self.model.train()
        t2 = time.time()
        print(f'12x256x256x8 nn 1000 train time: {t2 - t1:.2f}s')
        self.assertEqual(self.model._train_steps, 1000)

    def test_04_weights(self):
        weights = self.model.get_weights()
        self.model.set_weights(weights)
        self.assertTrue('online' in weights)
        self.assertTrue('target' in weights)

    def test_05_buffer(self):
        buffer = self.model.get_buffer()
        self.model.set_buffer(buffer)
        self.assertEqual(buffer['size'], 1000)
        self.assertEqual(buffer['data']['acts_buf'].shape, (1000, 2, 8))

    def test_06_env_test(self):
        episode_reward_history = []
        loss_history = []
        episode_reward_mean = 0
        loss_mean = 0
        env = TwoStepEnv()
        init_state = tf.one_hot(0, self.model.obs_dim)
        terminated = False
        truncated = False
        writer = tf.summary.create_file_writer('board/qmix_logs')
        with tqdm.trange(516) as t:
            for episode in t:
                for agent in self.model.agent_list:
                    agent.reset(init_state)
                env.reset()
                rewards = []
                for step in range(20):
                    actions = []
                    for agent in self.model.agent_list:
                        action = agent.act()
                        actions.append(action)
                    state, reward, done = env.step(actions)
                    state = tf.one_hot(state, self.model.obs_dim)
                    rewards.append(reward)

                    observations = []
                    for agent in self.model.agent_list:
                        agent.observe(state)
                        trajectory = copy.deepcopy(agent.trajectory)
                        observations.append(trajectory)

                    one_hot_actions = []
                    for action in actions:
                        action = tf.one_hot(action, depth=self.model.act_num)
                        one_hot_actions.append(action)

                    if self.model.prev_state is None:
                        self.model.prev_state = state
                        self.model.prev_observations = observations
                    self.model.store(self.model.prev_state, one_hot_actions, state, reward, terminated, truncated)
                    self.model.prev_obs1_list.append(self.model.prev_observations)
                    self.model.prev_obs2_list.append(observations)
                    self.model.prev_state = state
                    self.model.prev_observations = observations
                    # if (episode+1) % 20 == 0:
                    loss = self.model.train()
                    loss_history.append(loss)
                    # print(loss)
                episode_reward = np.sum(rewards)
                with writer.as_default():
                    tf.summary.scalar("score", episode_reward, 20 * (episode + 1))
                episode_reward_history.append(episode_reward)
                episode_reward_mean = 0.01 * episode_reward + 0.99 * episode_reward_mean
                t.set_description(
                    f"Episode:{episode},state:{env.prev_state}, reward:{episode_reward}")
                t.set_postfix(episode_reward_mean=episode_reward_mean)

        # fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
        # axL.plot(
        #     np.arange(
        #         len(episode_reward_history)),
        #     episode_reward_history,
        #     label="episode_reward")
        # axL.set_xlabel('episode')
        # axL.set_title("episode reward history")
        #
        # axR.plot(np.arange(len(loss_history)), loss_history, label="loss")
        # axR.set_title("qmix's loss history")
        #
        # axR.legend()
        # axL.legend()
        # plt.savefig("result.png")


    # def test_06_status(self):
    #     status = self.model.get_status()
    #     self.model.set_status(status)
    #     self.assertEqual(status['react_steps'], 100)
    #     self.assertEqual(status['train_steps'], 100)
    #     self.assertEqual(status['states_dim'], 12)
    #     self.assertEqual(status['actions_num'], 8)
