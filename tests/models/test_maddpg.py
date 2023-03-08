import json
import time
import unittest
import numpy as np
from models.maddpg import MADDPG


class MADDPGModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('tests/models/test_maddpg_src/hypers.json', 'r') as f:
            hypers = json.load(f)
        cls.model = MADDPG(training=True, **hypers['hypers'])

    # @classmethod
    # def tearDownClass(cls):
    #     cls.model.close()
    #     cls.model = None

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
        # self.assertIsInstance(action_n[0],np.ndarray)
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
        self.assertTrue('actor_target' in weights)

    def test_05_buffer(self):
        buffer = self.model.get_buffer()
        self.model.set_buffer(buffer)
        self.assertEqual(buffer['size'], 1000)
        self.assertEqual(buffer['data']['acts_buf'].shape, (1000, 8))

    # def test_06_status(self):
    #     status = self.model.get_status()
    #     self.model.set_status(status)
    #     self.assertEqual(status['react_steps'], 100)
    #     self.assertEqual(status['train_steps'], 100)
    #     self.assertEqual(status['states_dim'], 12)
    #     self.assertEqual(status['actions_num'], 8)
