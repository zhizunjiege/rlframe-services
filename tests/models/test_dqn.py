import json
import time
import unittest

import numpy as np

from models.dqn import DQN
from models.utils import default_builder


class DQNModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('examples/hypers.json', 'r') as f1, \
             open('examples/structures.json', 'r') as f2:
            hypers = json.loads(f1.read())
            structures = json.loads(f2.read())
        cls.model = DQN(
            training=True,
            networks=default_builder(structures),
            **hypers,
        )

    @classmethod
    def tearDownClass(cls):
        cls.model = None

    def test_00_init(self):
        self.assertTrue(self.model.training)

    def test_01_react(self):
        states = np.random.random((20,))
        t1 = time.time()
        for _ in range(5000):
            action = self.model.react(states)
        t2 = time.time()
        print(f'20x256x256x8 nn 5000 react time: {t2 - t1:.2f}s')
        self.assertEqual(type(action), int)

    def test_02_store(self):
        states = np.random.random((20,))
        action = 0
        next_states = np.random.random((20,))
        reward = 0.0
        done = False
        for _ in range(1000):
            self.model.store(states, action, next_states, reward, done)
        self.assertEqual(self.model.replay_buffer.size, 1000)

    def test_03_train(self):
        t1 = time.time()
        for _ in range(5000):
            self.model.train()
        t2 = time.time()
        print(f'20x256x256x8 nn 5000 train time: {t2 - t1:.2f}s')
        self.assertEqual(self.model._train_steps, 5000)

    def test_04_weight(self):
        weights = self.model.get_weights()
        self.model.set_weights(weights)
        self.assertTrue('online' in weights)
        self.assertTrue('target' in weights)

    def test_05_buffer(self):
        buffer = self.model.get_buffer()
        self.model.set_buffer(buffer)
        self.assertEqual(buffer['size'], 1000)
        self.assertEqual(buffer['data']['acts_buf'].shape, (1000, 1))

    def test_06_status(self):
        status = self.model.get_status()
        self.model.set_status(status)
        self.assertEqual(status['react_steps'], 5000)
        self.assertEqual(status['train_steps'], 5000)
        self.assertEqual(status['states_dim'], 20)
        self.assertEqual(status['actions_num'], 8)
