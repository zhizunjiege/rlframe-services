import unittest

import numpy as np

from models.buffer.single import SingleAgentBuffer


class SingleAgentBufferTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ...

    @classmethod
    def tearDownClass(cls):
        ...

    def test_00_case1(self):
        buffer = SingleAgentBuffer(obs_dim=4, act_dim=1, max_size=200)
        self.assertEqual(buffer.obs1_buf.shape, (200, 4))
        self.assertEqual(buffer.acts_buf.shape, (200, 1))
        self.assertEqual(buffer.rews_buf.shape, (200, 1))
        self.assertEqual(buffer.term_buf.shape, (200, 1))

        self.assertEqual(buffer.ptr, 0)
        self.assertEqual(buffer.size, 0)

        obs = np.array([1, 2, 3, 4])
        act = np.array([0])
        next_obs = np.array([5, 6, 7, 8])
        rew = 0.0
        term = False
        for _ in range(100):
            rew += 1
            buffer.store(obs, act, next_obs, rew, term)

        state = buffer.get()
        buffer.set(state)

        self.assertEqual(buffer.ptr, 100)
        self.assertEqual(buffer.size, 100)

        idx = np.random.randint(0, 100)
        self.assertEqual(buffer.obs1_buf[idx].shape, (4,))
        self.assertEqual(buffer.acts_buf[idx].shape, (1,))
        self.assertEqual(buffer.rews_buf[idx].shape, (1,))
        self.assertEqual(buffer.term_buf[idx].shape, (1,))

        batch = buffer.sample(10)
        self.assertEqual(batch['obs1'].shape, (10, 4))
        self.assertEqual(batch['acts'].shape, (10, 1))
        self.assertEqual(batch['rews'].shape, (10, 1))
        self.assertEqual(batch['term'].shape, (10, 1))

        # print(batch)

    def test_00_case2(self):
        buffer = SingleAgentBuffer(obs_dim=4, act_dim=2, max_size=100)
        self.assertEqual(buffer.obs1_buf.shape, (100, 4))
        self.assertEqual(buffer.acts_buf.shape, (100, 2))
        self.assertEqual(buffer.rews_buf.shape, (100, 1))
        self.assertEqual(buffer.term_buf.shape, (100, 1))

        self.assertEqual(buffer.ptr, 0)
        self.assertEqual(buffer.size, 0)

        obs = np.array([1, 2, 3, 4])
        act = np.array([0, 1])
        next_obs = np.array([5, 6, 7, 8])
        rew = 0.0
        term = True
        for _ in range(101):
            rew += 1
            buffer.store(obs, act, next_obs, rew, term)

        state = buffer.get()
        buffer.set(state)

        self.assertEqual(buffer.ptr, 101)
        self.assertEqual(buffer.size, 100)
        self.assertAlmostEqual(float(buffer.rews_buf[0, 0]), 101.0)

        idx = np.random.randint(0, 100)
        self.assertEqual(buffer.obs1_buf[idx].shape, (4,))
        self.assertEqual(buffer.acts_buf[idx].shape, (2,))
        self.assertEqual(buffer.rews_buf[idx].shape, (1,))
        self.assertEqual(buffer.term_buf[idx].shape, (1,))

        batch = buffer.sample(10)
        self.assertEqual(batch['obs1'].shape, (10, 4))
        self.assertEqual(batch['acts'].shape, (10, 2))
        self.assertEqual(batch['rews'].shape, (10, 1))
        self.assertEqual(batch['term'].shape, (10, 1))

        # print(batch)
