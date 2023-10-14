import time
import unittest

import numpy as np

from models.doubledqn import DoubleDQN


class DoubleDQNModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DoubleDQN(
            training=True,
            obs_dim=12,
            act_num=8,
            hidden_layers=[64, 64],
            lr=0.001,
            gamma=0.99,
            buffer_size=1000000,
            batch_size=64,
            epsilon_max=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.9,
            start_steps=0,
            update_after=64,
            update_online_every=1,
            update_target_every=200,
            seed=None,
        )

    @classmethod
    def tearDownClass(cls):
        cls.model = None

    def test_00_init(self):
        self.assertTrue(self.model.training)

    def test_01_react(self):
        states = np.random.random((12,))
        t1 = time.time()
        for _ in range(5000):
            action = self.model.react(states)
        t2 = time.time()
        print(f'12x64x64x8 nn 5000 react time: {t2 - t1:.2f}s')
        self.assertIsInstance(action, int)

    def test_02_store(self):
        states = np.random.random((12,))
        action = 0
        next_states = np.random.random((12,))
        reward = 0.0
        terminated = False
        truncated = False
        for _ in range(1000):
            self.model.store(states, action, next_states, reward, terminated, truncated)
        self.assertEqual(self.model.buffer.size, 1000)

    def test_03_train(self):
        t1 = time.time()
        for _ in range(5000):
            self.model.train()
        t2 = time.time()
        print(f'12x64x64x8 nn 5000 train time: {t2 - t1:.2f}s')
        self.assertEqual(self.model._train_steps, 5000)

    def test_04_weights(self):
        weights = self.model.get_weights()
        self.model.set_weights(weights)
        self.assertTrue('online' in weights)
        self.assertTrue('target' in weights)

    def test_05_buffer(self):
        buffer = self.model.get_buffer()
        self.model.set_buffer(buffer)
        self.assertEqual(buffer['ptr'], 1000)
        self.assertEqual(buffer['acts'].shape, (1000, 1))

    def test_06_status(self):
        status = self.model.get_status()
        self.model.set_status(status)
        self.assertEqual(status['react_steps'], 5000)
        self.assertEqual(status['train_steps'], 5000)

    def test_07_gymnasium(self):
        from datetime import datetime
        import logging
        import os

        logdir = f'tests/logs/models/doubledqn/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        os.makedirs(logdir, exist_ok=True)

        handler = logging.FileHandler(f'{logdir}/log.txt', mode="w", encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger = logging.getLogger('DoubleDQN')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        import gymnasium as gym

        NAME = 'MountainCar-v0'
        SEED = 0
        STEP = 50000

        env = gym.wrappers.TimeLimit(gym.make(NAME).unwrapped, max_episode_steps=STEP)
        model = DoubleDQN(
            training=True,
            obs_dim=env.observation_space.shape[0],
            act_num=env.action_space.n,
            hidden_layers=[128, 128],
            lr=0.001,
            gamma=0.99,
            buffer_size=1000000,
            batch_size=64,
            epsilon_max=1.0,
            epsilon_min=0.0,
            epsilon_decay=0.99996,
            start_steps=STEP,
            update_after=STEP,
            update_online_every=1,
            update_target_every=200,
            seed=SEED,
        )

        returns = []
        for episode in range(1000):
            ret = 0
            states, _ = env.reset(seed=SEED)
            for step in range(STEP):
                actions = model.react(states=states)
                next_states, reward, terminated, truncated, _ = env.step(actions)
                model.store(states, actions, next_states, reward, terminated, truncated)
                model.train()
                ret += reward
                if terminated or truncated:
                    break
                else:
                    states = next_states

            logger.info(f'Training episode {episode} finished after {step+1} steps with return {ret:.2f}')
            returns.append(ret)

        import pandas as pd
        df = pd.DataFrame(returns, columns=['returns'])
        df.to_csv(f'{logdir}/returns.csv', index=False, header=True)

        env = gym.wrappers.TimeLimit(gym.make(NAME, render_mode='rgb_array').unwrapped, max_episode_steps=STEP)
        model.training = False

        frames = []
        ret = 0
        states, _ = env.reset(seed=SEED)
        for step in range(STEP):
            frames.append(env.render())
            actions = model.react(states=states)
            next_states, reward, terminated, truncated, _ = env.step(actions)
            ret += reward
            if terminated or truncated:
                break
            else:
                states = next_states

        logger.info(f'Testing episode finished after {step+1} steps with return {ret:.2f}')

        import imageio
        imageio.mimsave(f'{logdir}/{NAME}.gif', frames, 'GIF', duration=40, loop=0)
