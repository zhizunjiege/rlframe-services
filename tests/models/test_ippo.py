import json
import time
import unittest
from typing import Union, Dict, Optional
import numpy as np
from matplotlib import pyplot as plt
from models.ippo import IPPO
from models.mpe_env import mpe_env
import tensorflow as tf


class IPPOModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # with open('G:/RL/RLFrame-main/tests/models/test_maddpg_src/hypers.json', 'r') as f1, \
        #      open('G:/RL/RLFrame-main/tests/models/test_maddpg_src/structs.json', 'r') as f2:
        #     hypers = json.load(f1)
        #     structs = json.load(f2)
        # cls.model = MADDPG(training=True, networks=default_builder(structs), **hypers['hypers'])
        with open('tests/models/test_ippo_src/hypers.json', 'r') as f:
            hypers = json.load(f)
        cls.model = IPPO(training=True, **hypers['hypers'])

    @classmethod
    def tearDownClass(cls):
        cls.model.close()
        cls.model = None

    def test_00_init(self):
        self.assertTrue(self.model.training)
        # self.assertEqual(self.model.__nobs, 12)
        # self.assertEqual(self.model.__nact, 8)

    def test_01_react(self):
        states = {}
        for agent_index in range(self.model.agent_num):
            states[agent_index] = np.random.random((12,))
        t1 = time.time()
        for _ in range(1000):
            action_n = self.model.react(states)
        t2 = time.time()
        print(f'12x256x256x8 nn 1000 react time: {t2 - t1:.2f}s')
        print(states[0].shape)
        self.assertIsInstance(action_n[0], np.ndarray)
        print(action_n[0].shape)
        self.assertEqual(action_n[0].shape[0], 8)

    def test_02_store(self):
        states = {}
        action = {}
        next_states = {}
        reward = {}
        terminated = False
        truncated = False
        for agent_index in range(self.model.agent_num):
            states[agent_index] = np.random.random((12,))
            action[agent_index] = np.random.random((8,))
            next_states[agent_index] = np.random.random((12,))
            reward[agent_index] = 0.0
        for _ in range(1000):
            self.model.store(states, action, next_states, reward, terminated, truncated)
        self.assertEqual(self.model.replay_buffer_list[0].size, 1000)

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
        self.assertTrue('actor' in weights)
        self.assertTrue('actor_old' in weights)

    def test_05_buffer(self):
        buffer = self.model.get_buffer()
        self.model.set_buffer(buffer)
        self.assertEqual(buffer['size'], 1000)
        self.assertEqual(buffer['data']['acts_buf'].shape, (1000, 8))

    def test_06_gymnasium(self):
        SEED = 0
        env = mpe_env('simple_spread', seed=SEED)
        obs_dim, action_dim = env.get_space()
        agent_number = env.get_agent_number()
        # policy初始化
        ippo_agents = IPPO(
            training=True,
            obs_dim=obs_dim,
            act_num=action_dim,
            hidden_layers=[64, 64],
            actor_lr=0.02,
            critic_lr=0.02,
            gamma=0.95,
            replay_size=10000,
            batch_size=32,
            start_steps=0,
            update_after=20,
            update_online_every=1,
            update_target_every=20,
            seed=SEED,
            agent_num=agent_number,
            noise_range=0.1,
            action_bound=0.5,
            epsilon=0.15
        )
        score = []
        avg_score = []
        terminated = False
        truncated = False
        writer = tf.summary.create_file_writer('board/ippo_logs')
        for i_episode in range(1000):
            obs_n = env.mpe_env.reset()
            score_one_episode = 0
            # 20是单个回合内的步数
            for t in range(20):
                env.mpe_env.render()
                action_n = ippo_agents.react(obs_n)
                # action_n = [np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 0])]
                new_obs_n, reward_n, done_n, info_n = env.mpe_env.step(list(action_n.values()))
                rew_n = {}
                for idx in range(agent_number):
                    rew_n[idx] = reward_n[idx][0]
                ippo_agents.store(obs_n, action_n, new_obs_n, rew_n, terminated, truncated)
                loss = ippo_agents.train()
                # for agent_index in range(agent_number):
                #     score_one_episode += reward_n[agent_index][0]
                score_one_episode += reward_n[0][0]
                obs_n = new_obs_n
            print(loss)
            with writer.as_default():
                tf.summary.scalar("score", score_one_episode, 20 * (i_episode + 1))
            if (i_episode + 1) % 10 == 0:
                plt.plot(score)  # 绘制波形
                plt.plot(avg_score)  # 绘制波形
                # plt.show()
            score.append(score_one_episode)
            avg = np.mean(score[-100:])
            avg_score.append(avg)
            print(f"i_episode is {i_episode},score_one_episode is {score_one_episode},avg_score is {avg}")
        env.mpe_env.close()
    # def test_06_status(self):
    #     status = self.model.get_status()
    #     self.model.set_status(status)
    #     self.assertEqual(status['react_steps'], 100)
    #     self.assertEqual(status['train_steps'], 100)
    #     self.assertEqual(status['states_dim'], 12)
    #     self.assertEqual(status['actions_num'], 8)
