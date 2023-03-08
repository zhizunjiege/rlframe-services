import json
import time
import unittest

import numpy as np

from models.ddpg import DDPG
from models.utils import default_builder

class DDPGModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # with open('G:/RL/RLFrame-main/tests/models/test_ddpg_src/hypers.json', 'r') as f1, \
        #      open('G:/RL/RLFrame-main/tests/models/test_ddpg_src/structs.json', 'r') as f2:
        #     hypers = json.load(f1)
        #     structs = json.load(f2)
        # cls.model = DDPG(training=True, networks=default_builder(structs), **hypers['hypers'])
        with open('tests/models/test_ddpg_src/hypers.json', 'r') as f:
            hypers = json.load(f)
        cls.model = DDPG(training=True, **hypers['hypers'])
    # @classmethod
    # def tearDownClass(cls):
    #     cls.model.close()
    #     cls.model = None

    def test_00_init(self):
        self.assertTrue(self.model.training)

    def test_01_react(self):
        states = np.random.random((12,))
        t1 = time.time()
        for _ in range(100):
            action = self.model.react(states)
        t2 = time.time()
        print(f'12x256x256x8 nn 5000 react time: {t2 - t1:.2f}s')
        print(states.shape)
        self.assertIsInstance(action, np.ndarray)
        print(action.shape)
        self.assertEqual(action.shape[0], 8)

    def test_02_store(self):
        states = np.random.random((12,))
        action = 0
        next_states = np.random.random((12,))
        reward = 0.0
        terminated = False
        truncated = False
        for _ in range(1000):
            self.model.store(states, action, next_states, reward, terminated, truncated)
        self.assertEqual(self.model.replay_buffer.size, 1000)

    def test_03_train(self):
        t1 = time.time()
        for _ in range(100):
            self.model.train()
        t2 = time.time()
        print(f'12x256x256x8 nn 5000 train time: {t2 - t1:.2f}s')
        self.assertEqual(self.model._train_steps, 100)

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
