import unittest

import numpy as np

from models.replay.multi_replay import MultiReplay


class MultiReplayTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ...

    @classmethod
    def tearDownClass(cls):
        ...

    def test_00_case1(self):
        obs = [np.array([1, 2, 3, 4])] * 2
        act = [np.array([0])] * 2
        next_obs = [np.array([5, 6, 7, 8])] * 2
        rew = [0.0] * 2
        term = False

        replay = MultiReplay(agent_num=2, obs_dim=4, act_dim=1, max_size=1000, dtype='float32')
        self.assertEqual(len(replay.obs1_bufs), 2)
        self.assertEqual(len(replay.acts_bufs), 2)
        self.assertEqual(replay.obs1_bufs[0].shape, (1000, 4))
        self.assertEqual(replay.acts_bufs[0].shape, (1000, 1))
        self.assertEqual(replay.rews_buf.shape, (2, 1000))
        self.assertEqual(replay.term_buf.shape, (1000,))

        self.assertEqual(replay.ptr, 0)
        self.assertEqual(replay.size, 0)

        for _ in range(100):
            replay.store(obs, act, next_obs, rew, term)

        idx = np.random.randint(0, 100)
        self.assertEqual(replay.obs1_bufs[0][idx].shape, (4,))
        self.assertEqual(replay.acts_bufs[0][idx].shape, (1,))
        self.assertEqual(replay.rews_buf[:, idx].shape, (2,))
        self.assertEqual(replay.term_buf[idx].shape, ())

        self.assertEqual(replay.ptr, 100)
        self.assertEqual(replay.size, 100)

        batch = replay.sample(10)
        self.assertEqual(len(batch['obs1']), 2)
        self.assertEqual(len(batch['acts']), 2)
        self.assertEqual(len(batch['rews']), 2)
        self.assertEqual(batch['obs1'][0].shape, (10, 4))
        self.assertEqual(batch['acts'][0].shape, (10, 1))
        self.assertEqual(batch['rews'][0].shape, (10,))
        self.assertEqual(batch['term'].shape, (10,))

        state = replay.get()
        replay.set(state)

        print(batch)

    def test_00_case2(self):
        obs = [np.array([1, 2, 3]), np.array([1, 2, 3, 4])]
        act = [np.array([0]), np.array([0, 1])]
        next_obs = [np.array([5, 6, 7]), np.array([5, 6, 7, 8])]
        rew = 1.0
        term = False

        replay = MultiReplay(agent_num=2, obs_dim=[3, 4], act_dim=[1, 2], max_size=1000, dtype='float32')
        self.assertEqual(len(replay.obs1_bufs), 2)
        self.assertEqual(len(replay.acts_bufs), 2)
        self.assertEqual(replay.obs1_bufs[0].shape, (1000, 3))
        self.assertEqual(replay.obs1_bufs[1].shape, (1000, 4))
        self.assertEqual(replay.acts_bufs[0].shape, (1000, 1))
        self.assertEqual(replay.acts_bufs[1].shape, (1000, 2))
        self.assertEqual(replay.rews_buf.shape, (2, 1000))
        self.assertEqual(replay.term_buf.shape, (1000,))

        self.assertEqual(replay.ptr, 0)
        self.assertEqual(replay.size, 0)

        for _ in range(100):
            replay.store(obs, act, next_obs, rew, term)

        idx = np.random.randint(0, 100)
        self.assertEqual(replay.obs1_bufs[0][idx].shape, (3,))
        self.assertEqual(replay.obs1_bufs[1][idx].shape, (4,))
        self.assertEqual(replay.acts_bufs[0][idx].shape, (1,))
        self.assertEqual(replay.acts_bufs[1][idx].shape, (2,))
        self.assertEqual(replay.rews_buf[:, idx].shape, (2,))
        self.assertEqual(replay.term_buf[idx].shape, ())

        self.assertEqual(replay.ptr, 100)
        self.assertEqual(replay.size, 100)

        batch = replay.sample(10)
        self.assertEqual(len(batch['obs1']), 2)
        self.assertEqual(len(batch['acts']), 2)
        self.assertEqual(len(batch['rews']), 2)
        self.assertEqual(batch['obs1'][0].shape, (10, 3))
        self.assertEqual(batch['obs1'][1].shape, (10, 4))
        self.assertEqual(batch['acts'][0].shape, (10, 1))
        self.assertEqual(batch['acts'][1].shape, (10, 2))
        self.assertEqual(batch['rews'][0].shape, (10,))
        self.assertEqual(batch['rews'][1].shape, (10,))
        self.assertEqual(batch['term'].shape, (10,))

        state = replay.get()
        replay.set(state)

        print(batch)
