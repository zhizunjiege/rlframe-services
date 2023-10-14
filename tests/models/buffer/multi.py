import unittest

import numpy as np

from models.buffer.multi import MultiAgentBuffer


class MultiAgentBufferTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ...

    @classmethod
    def tearDownClass(cls):
        ...

    def test_00_case1(self):
        buffer = MultiAgentBuffer(number=2, obs_dim=4, act_dim=1, max_size=200)
        self.assertEqual(len(buffer.obs1_bufs), 2)
        self.assertEqual(len(buffer.acts_bufs), 2)
        self.assertEqual(len(buffer.rews_bufs), 2)
        self.assertIsInstance(buffer.term_buf, np.ndarray)
        self.assertEqual(buffer.obs1_bufs[0].shape, (200, 4))
        self.assertEqual(buffer.acts_bufs[0].shape, (200, 1))
        self.assertEqual(buffer.rews_bufs[0].shape, (200, 1))
        self.assertEqual(buffer.term_buf.shape, (200, 1))

        self.assertEqual(buffer.ptr, 0)
        self.assertEqual(buffer.size, 0)

        obs = [np.array([1, 2, 3, 4])] * 2
        act = [np.array([0])] * 2
        next_obs = [np.array([5, 6, 7, 8])] * 2
        rew = [0.0] * 2
        term = False
        for _ in range(100):
            buffer.store(obs, act, next_obs, rew, term)

        state = buffer.get()
        buffer.set(state)

        self.assertEqual(buffer.ptr, 100)
        self.assertEqual(buffer.size, 100)

        idx = np.random.randint(0, 100)
        self.assertEqual(buffer.obs1_bufs[0][idx].shape, (4,))
        self.assertEqual(buffer.acts_bufs[0][idx].shape, (1,))
        self.assertEqual(buffer.rews_bufs[0][idx].shape, (1,))
        self.assertEqual(buffer.term_buf[idx].shape, (1,))

        batch = buffer.sample(10)
        self.assertEqual(len(batch['obs1']), 2)
        self.assertEqual(len(batch['acts']), 2)
        self.assertEqual(len(batch['rews']), 2)
        self.assertIsInstance(batch['term'], np.ndarray)
        self.assertEqual(batch['obs1'][0].shape, (10, 4))
        self.assertEqual(batch['acts'][0].shape, (10, 1))
        self.assertEqual(batch['rews'][0].shape, (10, 1))
        self.assertEqual(batch['term'].shape, (10, 1))

        # print(batch)

    def test_00_case2(self):
        buffer = MultiAgentBuffer(number=2, obs_dim=[3, 4], act_dim=[1, 2], max_size=100)
        self.assertEqual(len(buffer.obs1_bufs), 2)
        self.assertEqual(len(buffer.acts_bufs), 2)
        self.assertEqual(len(buffer.rews_bufs), 2)
        self.assertIsInstance(buffer.term_buf, np.ndarray)
        self.assertEqual(buffer.obs1_bufs[0].shape, (100, 3))
        self.assertEqual(buffer.obs1_bufs[1].shape, (100, 4))
        self.assertEqual(buffer.acts_bufs[0].shape, (100, 1))
        self.assertEqual(buffer.acts_bufs[1].shape, (100, 2))
        self.assertEqual(buffer.rews_bufs[0].shape, (100, 1))
        self.assertEqual(buffer.term_buf.shape, (100, 1))

        self.assertEqual(buffer.ptr, 0)
        self.assertEqual(buffer.size, 0)

        obs = [np.array([1, 2, 3]), np.array([1, 2, 3, 4])]
        act = [np.array([0]), np.array([0, 1])]
        next_obs = [np.array([5, 6, 7]), np.array([5, 6, 7, 8])]
        rew = [0.0] * 2
        term = True
        for _ in range(101):
            for i in range(2):
                rew[i] += 1
            buffer.store(obs, act, next_obs, rew, term)

        state = buffer.get()
        buffer.set(state)

        self.assertEqual(buffer.ptr, 101)
        self.assertEqual(buffer.size, 100)
        self.assertAlmostEqual(float(buffer.rews_bufs[0][0, 0]), 101.0)

        idx = np.random.randint(0, 100)
        self.assertEqual(buffer.obs1_bufs[0][idx].shape, (3,))
        self.assertEqual(buffer.obs1_bufs[1][idx].shape, (4,))
        self.assertEqual(buffer.acts_bufs[0][idx].shape, (1,))
        self.assertEqual(buffer.acts_bufs[1][idx].shape, (2,))
        self.assertEqual(buffer.rews_bufs[0][idx].shape, (1,))
        self.assertEqual(buffer.term_buf[idx].shape, (1,))

        batch = buffer.sample(10)
        self.assertEqual(len(batch['obs1']), 2)
        self.assertEqual(len(batch['acts']), 2)
        self.assertEqual(len(batch['rews']), 2)
        self.assertIsInstance(batch['term'], np.ndarray)
        self.assertEqual(batch['obs1'][0].shape, (10, 3))
        self.assertEqual(batch['obs1'][1].shape, (10, 4))
        self.assertEqual(batch['acts'][0].shape, (10, 1))
        self.assertEqual(batch['acts'][1].shape, (10, 2))
        self.assertEqual(batch['rews'][0].shape, (10, 1))
        self.assertEqual(batch['term'].shape, (10, 1))

        # print(batch)
