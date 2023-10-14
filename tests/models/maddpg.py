import time
import unittest

import numpy as np

from models.maddpg import MADDPG


class MADDPGModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = MADDPG(
            training=True,
            number=2,
            obs_dim=12,
            act_dim=8,
            hidden_layers_actor=[64, 64],
            hidden_layers_critic=[64, 64],
            lr_actor=0.0001,
            lr_critic=0.001,
            gamma=0.99,
            tau=0.001,
            buffer_size=1000000,
            batch_size=64,
            noise_type='ou',
            noise_sigma=0.2,
            noise_theta=0.15,
            noise_dt=0.01,
            noise_max=1.0,
            noise_min=1.0,
            noise_decay=1.0,
            update_after=64,
            update_every=1,
            seed=None,
        )

    @classmethod
    def tearDownClass(cls):
        cls.model = None

    def test_00_init(self):
        self.assertTrue(self.model.training)

    def test_01_react(self):
        states = {i: np.random.random((12,)) for i in range(self.model.number)}
        t1 = time.time()
        for _ in range(5000):
            actions = self.model.react(states)
        t2 = time.time()
        print(f'2x12x64x64x8 nn 5000 react time: {t2 - t1:.2f}s')
        self.assertIsInstance(actions, dict)
        self.assertIsInstance(actions[0], np.ndarray)
        self.assertEqual(actions[0].shape[0], 8)

    def test_02_store(self):
        states = {i: np.random.random((12,)) for i in range(self.model.number)}
        actions = {i: np.random.random((8,)) for i in range(self.model.number)}
        next_states = {i: np.random.random((12,)) for i in range(self.model.number)}
        reward = {i: 0.0 for i in range(self.model.number)}
        terminated = False
        truncated = False
        for _ in range(1000):
            self.model.store(states, actions, next_states, reward, terminated, truncated)
        self.assertEqual(self.model.buffer.size, 1000)

    def test_03_train(self):
        t1 = time.time()
        for _ in range(5000):
            self.model.train()
        t2 = time.time()
        print(f'2x12x64x64x8 nn 5000 train time: {t2 - t1:.2f}s')
        self.assertEqual(self.model._train_steps, 5000)

    def test_04_weights(self):
        weights = self.model.get_weights()
        self.model.set_weights(weights)
        self.assertTrue('actor' in weights)
        self.assertTrue('critic' in weights)
        self.assertTrue('actor_target' in weights)
        self.assertTrue('critic_target' in weights)

    def test_05_buffer(self):
        buffer = self.model.get_buffer()
        self.model.set_buffer(buffer)
        self.assertEqual(buffer['ptr'], 1000)
        self.assertEqual(buffer['acts'][0].shape, (1000, 8))

    def test_06_status(self):
        status = self.model.get_status()
        self.model.set_status(status)
        self.assertEqual(status['react_steps'], 5000)
        self.assertEqual(status['train_steps'], 5000)

    def test_07_pettingzoo(self):
        from datetime import datetime
        import logging
        import os

        logdir = f'tests/logs/models/maddpg/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        os.makedirs(logdir, exist_ok=True)

        handler = logging.FileHandler(f'{logdir}/log.txt', mode="w", encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger = logging.getLogger('MADDPG')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        from pettingzoo.mpe import simple_spread_v3

        def strkey2intkey(dict, keys):
            return {idx: dict[key] for idx, key in enumerate(keys)}

        def intkey2strkey(dict, keys):
            return {key: dict[idx] for idx, key in enumerate(keys)}

        def avg(array):
            return sum(array) / len(array)

        NAME = 'simple_spread_v3'
        SEED = 0
        STEP = 25

        env = simple_spread_v3.parallel_env(N=2, continuous_actions=True)
        env.reset(seed=SEED)
        model = MADDPG(
            training=True,
            number=env.max_num_agents,
            obs_dim=env.observation_space(env.agents[0]).shape[0],
            act_dim=env.action_space(env.agents[0]).shape[0],
            hidden_layers_actor=[128, 128],
            hidden_layers_critic=[128, 128],
            lr_actor=0.0003,
            lr_critic=0.003,
            gamma=0.98,
            tau=0.005,
            buffer_size=50000,
            batch_size=64,
            noise_type='normal',
            noise_sigma=0.1,
            noise_max=1.0,
            noise_min=0.1,
            noise_decay=0.9995,
            update_after=100,
            update_every=1,
            seed=SEED,
        )

        returns = []
        for episode in range(5000):
            ret = {k: 0.0 for k in env.possible_agents}
            obs_dict, _ = env.reset(seed=SEED)
            for step in range(STEP):
                obs = strkey2intkey(obs_dict, env.possible_agents)
                act = model.react(obs)
                act_dict = intkey2strkey({k: np.clip(0.5 * (v + 1), 0, 1) for k, v in act.items()}, env.possible_agents)
                next_obs_dict, rew_dict, term_dict, trun_dict, _ = env.step(act_dict)
                next_obs = strkey2intkey(next_obs_dict, env.possible_agents)
                rew = strkey2intkey(rew_dict, env.possible_agents)
                term = all(term_dict.values())
                trun = any(trun_dict.values())
                model.store(obs, act, next_obs, rew, term, trun)
                model.train()
                for k in env.possible_agents:
                    ret[k] += rew_dict[k]
                if term or trun:
                    break
                else:
                    obs_dict = next_obs_dict

            ret = avg(list(ret.values()))
            logger.info(f'Training episode {episode} finished after {step+1} steps with average return {ret:.2f}')
            returns.append(ret)

        import pandas as pd
        df = pd.DataFrame(returns, columns=['returns'])
        df.to_csv(f'{logdir}/returns.csv', index=False, header=True)

        env = simple_spread_v3.parallel_env(N=2, continuous_actions=True, render_mode='rgb_array')
        model.training = False

        frames = []
        ret = {k: 0.0 for k in env.possible_agents}
        obs_dict, _ = env.reset(seed=SEED)
        for step in range(STEP):
            frames.append(env.render())
            obs = strkey2intkey(obs_dict, env.possible_agents)
            act = model.react(obs)
            act_dict = intkey2strkey({k: np.clip(0.5 * (v + 1), 0, 1) for k, v in act.items()}, env.possible_agents)
            next_obs_dict, rew_dict, term_dict, trun_dict, _ = env.step(act_dict)
            term = all(term_dict.values())
            trun = any(trun_dict.values())
            for k in env.possible_agents:
                ret[k] += rew_dict[k]
            if term or trun:
                break
            else:
                obs_dict = next_obs_dict

        ret = avg(list(ret.values()))
        logger.info(f'Testing episode finished after {step+1} steps with average return {ret:.2f}')

        import imageio
        imageio.mimsave(f'{logdir}/{NAME}.gif', frames, 'GIF', duration=40, loop=0)
