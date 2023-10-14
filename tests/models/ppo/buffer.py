import unittest

import numpy as np

from models.ppo import PPOBuffer


class PPOBufferModelTestCase1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.buffer = PPOBuffer(obs_dim=4, act_dim=1, max_size=200, gamma=0.99, lam=0.95)

    @classmethod
    def tearDownClass(cls):
        ...

    def test_00_init(self):
        self.assertEqual(self.buffer.obs_buf.shape, (200, 4))
        self.assertEqual(self.buffer.act_buf.shape, (200, 1))
        self.assertEqual(self.buffer.ret_buf.shape, (200,))
        self.assertEqual(self.buffer.adv_buf.shape, (200,))
        self.assertEqual(self.buffer.pro_buf.shape, (200,))

        self.assertEqual(self.buffer.size, 0)
        self.assertFalse(self.buffer.full)

    def test_01_store(self):
        obs = np.array([1, 2, 3, 4])
        act = np.array([0])
        rew = 0.0
        val = 0.0
        pro = 0.5
        for _ in range(200):
            rew += 1.0
            val += 0.1
            self.buffer.store(obs, act, rew, val, pro)

        self.assertEqual(self.buffer.size, 200)

    def test_02_finish(self):
        self.assertEqual(self.buffer.path_start_idx, 0)
        self.assertEqual(self.buffer.ptr, 200)

        self.buffer.finish(last_val=0.0)

        self.assertEqual(self.buffer.path_start_idx, 200)
        self.assertEqual(self.buffer.ptr, 200)

        S = (1 - 0.99**200) / (1 - 0.99)**2 - 200 * 0.99**200 / (1 - 0.99)
        self.assertAlmostEqual(self.buffer.ret_buf[0], S, delta=0.1)

    def test_03_sample(self):
        data = self.buffer.sample()

        self.assertEqual(data['obs'].shape, (200, 4))
        self.assertEqual(data['act'].shape, (200, 1))
        self.assertEqual(data['ret'].shape, (200,))
        self.assertEqual(data['adv'].shape, (200,))
        self.assertEqual(data['pro'].shape, (200,))

        self.assertEqual(self.buffer.size, 0)

    def test_04_getset(self):
        state = self.buffer.get()
        self.buffer.set(state)

        self.assertEqual(self.buffer.ptr, 0)

    @unittest.expectedFailure
    def test_05_failure(self):
        obs = np.array([1, 2, 3, 4])
        act = np.array([0])
        rew = 0.0
        val = 0.0
        pro = 0.5
        for _ in range(100):
            rew += 1.0
            val += 0.1
            self.buffer.store(obs, act, rew, val, pro)

        self.buffer.finish(last_val=0.0)
        self.buffer.sample()


class PPOBufferbufferModelTestCase2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.buffer = PPOBuffer(obs_dim=12, act_dim=8, max_size=200, gamma=0.9, lam=0.9)

    @classmethod
    def tearDownClass(cls):
        ...

    def test_00_init(self):
        self.assertEqual(self.buffer.obs_buf.shape, (200, 12))
        self.assertEqual(self.buffer.act_buf.shape, (200, 8))
        self.assertEqual(self.buffer.ret_buf.shape, (200,))
        self.assertEqual(self.buffer.adv_buf.shape, (200,))
        self.assertEqual(self.buffer.pro_buf.shape, (200,))

        self.assertEqual(self.buffer.size, 0)
        self.assertFalse(self.buffer.full)

    def test_01_store(self):
        obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, -1, -2])
        act = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        rew = 0.1
        val = 1.0
        pro = 0.5
        for _ in range(200):
            rew += 1.0
            self.buffer.store(obs, act, rew, val, pro)

        self.assertEqual(self.buffer.size, 200)

    def test_02_finish(self):
        self.assertEqual(self.buffer.path_start_idx, 0)
        self.assertEqual(self.buffer.ptr, 200)

        self.buffer.finish(last_val=1.0)

        self.assertEqual(self.buffer.path_start_idx, 200)
        self.assertEqual(self.buffer.ptr, 200)

        S = (1 - 0.81**200) / (1 - 0.81)**2 - 200 * 0.81**200 / (1 - 0.81)
        self.assertAlmostEqual(self.buffer.adv_buf[0], S, delta=0.1)

    def test_03_sample(self):
        data = self.buffer.sample()

        self.assertEqual(data['obs'].shape, (200, 12))
        self.assertEqual(data['act'].shape, (200, 8))
        self.assertEqual(data['ret'].shape, (200,))
        self.assertEqual(data['adv'].shape, (200,))
        self.assertEqual(data['pro'].shape, (200,))

        self.assertEqual(self.buffer.size, 0)

    def test_04_getset(self):
        state = self.buffer.get()
        self.buffer.set(state)

        self.assertEqual(self.buffer.ptr, 0)

    @unittest.expectedFailure
    def test_05_failure(self):
        obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, -1, -2])
        act = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        rew = 0.0
        val = 0.0
        pro = 0.5
        for _ in range(100):
            self.buffer.store(obs, act, rew, val, pro)

        self.buffer.finish(last_val=0.0)
        self.buffer.sample()
