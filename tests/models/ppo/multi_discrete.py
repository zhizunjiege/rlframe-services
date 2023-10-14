import time
import unittest

import numpy as np

from models.ppo import PPO


class PPOModelMLPMultiDiscretePolicyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = PPO(
            training=True,
            policy='multi-discrete',
            obs_dim=12,
            act_dim=[2, 4, 8],
            hidden_layers_pi=[64, 64],
            hidden_layers_vf=[64, 64],
            lr_pi=0.0003,
            lr_vf=0.001,
            gamma=0.99,
            lam=0.97,
            epsilon=0.2,
            buffer_size=4000,
            update_pi_iter=80,
            update_vf_iter=80,
            max_kl=0.01,
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
        for _ in range(4000):
            actions = self.model.react(states)
        t2 = time.time()
        print(f'12x64x64x8 nn 4000 react time: {t2 - t1:.2f}s')
        self.assertIsInstance(actions, np.ndarray)
        self.assertEqual(actions.shape, (3,))

    def test_02_store(self):
        states = np.random.random((12,))
        actions = np.random.randint(0, 2, (3,))
        next_states = np.random.random((12,))
        reward = 0.0
        terminated = False
        truncated = False
        for _ in range(4000):
            self.model.store(states, actions, next_states, reward, terminated, truncated)
        self.assertEqual(self.model.buffer.size, 4000)

    def test_03_train(self):
        t1 = time.time()
        self.model.train()
        t2 = time.time()
        print(f'12x64x64x8 nn 1 train time: {t2 - t1:.2f}s')
        self.assertLessEqual(self.model._train_pi_steps, 80)
        self.assertEqual(self.model._train_vf_steps, 80)

    def test_04_weights(self):
        weights = self.model.get_weights()
        self.model.set_weights(weights)
        self.assertTrue('pi' in weights)
        self.assertTrue('vf' in weights)

    def test_05_buffer(self):
        buffer = self.model.get_buffer()
        self.model.set_buffer(buffer)
        self.assertEqual(buffer['ptr'], 0)
        self.assertEqual(buffer['act'].shape, (4000, 3))

    def test_06_status(self):
        status = self.model.get_status()
        self.model.set_status(status)
        self.assertEqual(status['react_steps'], 4000)
        self.assertLessEqual(status['train_pi_steps'], 80)
        self.assertEqual(status['train_vf_steps'], 80)

    def test_07_pettingzoo(self):
        from datetime import datetime
        import logging
        import os

        logdir = f'tests/logs/models/ppo/multi-discrete/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        os.makedirs(logdir, exist_ok=True)

        handler = logging.FileHandler(f'{logdir}/log.txt', mode="w", encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger = logging.getLogger('PPO')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        from pettingzoo.mpe import simple_spread_v3

        def dict2array(dict, keys):
            return np.concatenate([dict[key] for key in keys], axis=0)

        def array2dict(array, keys):
            return {key: item for key, item in zip(keys, np.split(array, len(keys), axis=0))}

        def avg(array):
            return sum(array) / len(array)

        NAME = 'simple_spread_v3'
        SEED = 0
        STEP = 25

        env = simple_spread_v3.parallel_env(N=2, continuous_actions=False)
        env.reset(seed=SEED)
        model = PPO(
            training=True,
            policy='multi-discrete',
            obs_dim=env.observation_space(env.agents[0]).shape[0] * env.max_num_agents,
            act_dim=[env.action_space(env.agents[0]).n] * env.max_num_agents,
            hidden_layers_pi=[128, 128],
            hidden_layers_vf=[128, 128],
            lr_pi=0.0003,
            lr_vf=0.003,
            gamma=0.98,
            lam=0.97,
            epsilon=0.2,
            buffer_size=200,
            update_pi_iter=100,
            update_vf_iter=100,
            max_kl=0.01,
            seed=SEED,
        )

        returns = []
        for episode in range(5000):
            ret = 0
            obs_dict, _ = env.reset(seed=SEED)
            for step in range(STEP):
                obs = dict2array(obs_dict, env.possible_agents)
                act = model.react(obs)
                act_dict = array2dict(act, env.possible_agents)
                act_dict = {key: int(value) for key, value in act_dict.items()}
                next_obs_dict, rew_dict, term_dict, trun_dict, _ = env.step(act_dict)
                next_obs = dict2array(next_obs_dict, env.possible_agents)
                rew = avg(rew_dict.values())
                term = all(term_dict.values())
                trun = any(trun_dict.values())
                model.store(obs, act, next_obs, rew, term, trun)
                model.train()
                ret += rew
                if term or trun:
                    break
                else:
                    obs_dict = next_obs_dict

            logger.info(f'Training episode {episode} finished after {step+1} steps with return {ret:.2f}')
            returns.append(ret)

        import pandas as pd
        df = pd.DataFrame(returns, columns=['returns'])
        df.to_csv(f'{logdir}/returns.csv', index=False, header=True)

        env = simple_spread_v3.parallel_env(N=2, continuous_actions=False, render_mode='rgb_array')
        model.training = False

        frames = []
        ret = 0
        obs_dict, _ = env.reset(seed=SEED)
        for step in range(STEP):
            frames.append(env.render())
            obs = dict2array(obs_dict, env.possible_agents)
            act = model.react(obs)
            act_dict = array2dict(act, env.possible_agents)
            act_dict = {key: int(value) for key, value in act_dict.items()}
            next_obs_dict, rew_dict, term_dict, trun_dict, _ = env.step(act_dict)
            rew = avg(rew_dict.values())
            term = all(term_dict.values())
            trun = any(trun_dict.values())
            ret += rew
            if term or trun:
                break
            else:
                obs_dict = next_obs_dict

        logger.info(f'Testing episode finished after {step+1} steps with return {ret:.2f}')

        import imageio
        imageio.mimsave(f'{logdir}/{NAME}.gif', frames, 'GIF', duration=40, loop=0)
