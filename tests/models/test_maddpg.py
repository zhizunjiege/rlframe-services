import json
import time
import unittest

import numpy as np
from pettingzoo.mpe import simple_spread_v2

from models.maddpg import MADDPG


class MADDPGModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('tests/models/test_maddpg_src/hypers.json', 'r') as f:
            hypers = json.load(f)
        cls.model = MADDPG(training=True, **hypers['hypers'])

    @classmethod
    def tearDownClass(cls):
        cls.model = None

    def test_00_init(self):
        self.assertTrue(self.model.training)

    def test_01_react(self):
        states = {}
        for agent_index in range(self.model.agent_num):
            states[agent_index] = np.random.random((12,))
        t1 = time.time()
        for _ in range(1000):
            action_n = self.model.react(states)
        t2 = time.time()
        print(f'12x256x256x8 nn 1000 react time: {t2 - t1:.2f}s')
        self.assertIsInstance(action_n[0], np.ndarray)
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
        self.assertEqual(self.model.replay_buffer.size, 1000)

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
        self.assertEqual(buffer['data']['acts_bufs'][0].shape, (1000, 8))

    def test_06_status(self):
        status = self.model.get_status()
        self.model.set_status(status)
        self.assertEqual(status['react_steps'], 1000)
        self.assertEqual(status['train_steps'], 1000)
        self.assertIn('noise_level', status)

    def test_07_gymnasium(self):

        def strkey2intkey(dict, keys):
            return {idx: dict[key] for idx, key in enumerate(keys)}

        def intkey2strkey(dict, keys):
            return {key: dict[idx] for idx, key in enumerate(keys)}

        SEED = 65535
        env = simple_spread_v2.parallel_env(continuous_actions=True, render_mode='human')
        env.reset(seed=SEED)

        agent_num = env.max_num_agents
        obs_dim = env.observation_space(env.agents[0]).shape[0]
        act_dim = env.action_space(env.agents[0]).shape[0]
        model = MADDPG(
            training=True,
            agent_num=agent_num,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_layers_actor=[64, 64],
            hidden_layers_critic=[64, 64],
            lr_actor=0.01,
            lr_critic=0.01,
            gamma=0.95,
            tau=0.01,
            replay_size=10000,
            batch_size=256,
            noise_type='normal',
            noise_sigma=0.1,
            noise_max=1.0,
            noise_min=1.0,
            noise_decay=1.0,
            update_after=20,
            update_online_every=1,
            seed=SEED,
        )

        for episode in range(800):
            obs_dict = env.reset(seed=SEED)
            obs_n = strkey2intkey(obs_dict, env.possible_agents)

            for step in range(25):
                act_n = model.react(obs_n)

                act_dict = intkey2strkey(act_n, env.possible_agents)
                next_obs_dict, rews_dict, term_dict, trun_dict, _ = env.step({k: 0.5 * (v + 1) for k, v in act_dict.items()})

                next_obs_n = strkey2intkey(next_obs_dict, env.possible_agents)
                rews_n = strkey2intkey(rews_dict, env.possible_agents)
                terminated = all(term_dict.values())
                truncated = any(trun_dict.values())

                model.store(obs_n, act_n, next_obs_n, rews_n, terminated, truncated)

                model.train()

                if terminated or truncated:
                    break
                else:
                    obs_dict = next_obs_dict
                    obs_n = next_obs_n

            print(f'Episode {episode} finished after {step+1} steps')

        env.close()
