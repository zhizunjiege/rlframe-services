import unittest

import numpy as np

from models.replay.single_replay import SingleReplay


class SingleReplayTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ...

    @classmethod
    def tearDownClass(cls):
        ...

    def test_00_case1(self):
        obs = np.array([1, 2, 3, 4])
        act = np.array([0])
        next_obs = np.array([5, 6, 7, 8])
        rew = 0.0
        term = False

        replay = SingleReplay(obs_dim=4, act_dim=1, max_size=1000, dtype='float32')
        self.assertEqual(replay.obs1_buf.shape, (1000, 4))
        self.assertEqual(replay.acts_buf.shape, (1000, 1))
        self.assertEqual(replay.rews_buf.shape, (1000,))
        self.assertEqual(replay.term_buf.shape, (1000,))

        self.assertEqual(replay.ptr, 0)
        self.assertEqual(replay.size, 0)

        for _ in range(100):
            replay.store(obs, act, next_obs, rew, term)

        idx = np.random.randint(0, 100)
        self.assertEqual(replay.obs1_buf[idx].shape, (4,))
        self.assertEqual(replay.acts_buf[idx].shape, (1,))
        self.assertEqual(replay.rews_buf[idx].shape, ())
        self.assertEqual(replay.term_buf[idx].shape, ())

        self.assertEqual(replay.ptr, 100)
        self.assertEqual(replay.size, 100)

        batch = replay.sample(10)
        self.assertEqual(batch['obs1'].shape, (10, 4))
        self.assertEqual(batch['acts'].shape, (10, 1))
        self.assertEqual(batch['rews'].shape, (10,))
        self.assertEqual(batch['term'].shape, (10,))

        state = replay.get()
        replay.set(state)

        print(batch)

    def test_00_case2(self):
        obs = np.array([1, 2, 3, 4])
        act = np.array([0, 1])
        next_obs = np.array([5, 6, 7, 8])
        rew = 1.0
        term = True

        replay = SingleReplay(obs_dim=4, act_dim=2, max_size=100, dtype='float16')
        self.assertEqual(replay.obs1_buf.shape, (100, 4))
        self.assertEqual(replay.acts_buf.shape, (100, 2))
        self.assertEqual(replay.rews_buf.shape, (100,))
        self.assertEqual(replay.term_buf.shape, (100,))

        self.assertEqual(replay.ptr, 0)
        self.assertEqual(replay.size, 0)

        for _ in range(101):
            replay.store(obs, act, next_obs, rew, term)

        idx = np.random.randint(0, 100)
        self.assertEqual(replay.obs1_buf[idx].shape, (4,))
        self.assertEqual(replay.acts_buf[idx].shape, (2,))
        self.assertEqual(replay.rews_buf[idx].shape, ())
        self.assertEqual(replay.term_buf[idx].shape, ())

        self.assertEqual(replay.ptr, 1)
        self.assertEqual(replay.size, 100)

        batch = replay.sample(10)
        self.assertEqual(batch['obs1'].shape, (10, 4))
        self.assertEqual(batch['acts'].shape, (10, 2))
        self.assertEqual(batch['rews'].shape, (10,))
        self.assertEqual(batch['term'].shape, (10,))

        state = replay.get()
        replay.set(state)

        print(batch)
