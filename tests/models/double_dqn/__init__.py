import json
import time
import unittest

import gymnasium as gym
import numpy as np

from models.double_dqn import DoubleDQN


class DoubleDQNModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('tests/models/double_dqn/hypers.json', 'r') as f:
            cls.hypers = json.load(f)
        cls.model = DoubleDQN(training=True, **cls.hypers)

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
        print(f'12x256x256x8 nn 5000 react time: {t2 - t1:.2f}s')
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
        self.assertEqual(self.model.replay_buffer.size, 1000)

    def test_03_train(self):
        t1 = time.time()
        for _ in range(5000):
            self.model.train()
        t2 = time.time()
        print(f'12x256x256x8 nn 5000 train time: {t2 - t1:.2f}s')
        self.assertEqual(self.model._train_steps, 5000)

    def test_04_weights(self):
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

    def test_07_gymnasium(self):
        env = gym.make('CartPole-v1')
        model = DoubleDQN(
            training=True,
            obs_dim=env.observation_space.shape[0],
            act_num=env.action_space.n,
            hidden_layers=[128],
            lr=0.01,
            gamma=0.99,
            replay_size=200000,
            batch_size=64,
            epsilon_max=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.9999,
            start_steps=5000,
            update_after=5000,
            update_online_every=1,
            update_target_every=200,
            seed=0,
        )

        count = 0
        for episode in range(1000):
            rew_sum = 0
            states, _ = env.reset(seed=0)
            for step in range(500):
                actions = model.react(states=states)
                next_states, reward, terminated, truncated, _ = env.step(actions)
                model.store(
                    states=states,
                    actions=actions,
                    next_states=next_states,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
                model.train()

                rew_sum += reward

                if terminated or truncated:
                    break
                else:
                    states = next_states

            print(f'Episode {episode} finished after {step+1} steps with reward {rew_sum:.2f}')

            if rew_sum >= 450:
                count += 1
            else:
                count = max(count - 1, 0)

            if count >= 10:
                break
