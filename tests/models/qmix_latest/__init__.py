import json
import time
import unittest
import numpy as np
import matplotlib.pyplot as plt
from models.qmix_latest.qmix import QMIX
from pettingzoo.mpe import simple_spread_v2

class QMIXModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('qmix_latest/hypers.json', 'r') as f:
            hypers = json.load(f)
        cls.model = QMIX(training=True, **hypers['hypers'])

    @classmethod
    def tearDownClass(cls):
        cls.model = None

    # def test_00_init(self):
    #     self.assertTrue(self.model.training)

    # def test_01_react(self):
    #     states = np.array([np.random.random((4,))] * self.model.agent_num)
    #     t1 = time.time()
    #     for _ in range(1000):
    #         action_n = self.model.react(states)
    #     t2 = time.time()
    #     print(f'12x256x256x8 nn 1000 react time: {t2 - t1:.2f}s')
    #     print(action_n[0])
    #     self.assertEqual(self.model._react_steps, 1000)
    #     # self.assertIsInstance(action_n[0], np.int32 or np.int64)

    # def test_02_store(self):
    #     states = np.array([np.random.random((4,))] * self.model.agent_num)
    #     action = np.array([[0] for _ in range(self.model.agent_num)])
    #     next_states = np.array([np.random.random((4,))] * self.model.agent_num)
    #     reward = np.array([1.0 for _ in range(self.model.agent_num)], dtype=np.float32)
    #     terminated = False
    #     for _ in range(1000):
    #         self.model.store(states, action, next_states, reward, terminated)
    #     self.assertEqual(self.model.replay_buffer.size, 1000)

    # def test_03_train(self):
    #     t1 = time.time()
    #     for _ in range(1000):
    #         self.model.train()
    #     t2 = time.time()
    #     print(f'12x256x256x8 nn 1000 train time: {t2 - t1:.2f}s')
    #     self.assertEqual(self.model._train_steps, 1000)

    # def test_04_weights(self):
    #     weights = self.model.get_weights()
    #     self.model.set_weights(weights)
    #     self.assertTrue('online' in weights)
    #     self.assertTrue('target' in weights)

    # def test_05_buffer(self):
    #     buffer = self.model.get_buffer()
    #     self.model.set_buffer(buffer)
    #     self.assertEqual(buffer['size'], 1000)

    # def test_06_status(self):
    #     status = self.model.get_status()
    #     self.model.set_status(status)
    #     self.assertEqual(status['react_steps'], 1000)
    #     self.assertEqual(status['train_steps'], 1000)
    #     self.assertIn('noise_level', status)

    def test_07_gymnasium(self):

        SEED = 65535
        env = simple_spread_v2.parallel_env(continuous_actions=False, render_mode='human')
        env.reset(seed=SEED)

        agent_num = env.max_num_agents
        print(agent_num)
        obs_dim = env.observation_space(env.agents[0]).shape[0]
        act_dim = env.action_space(env.agents[0]).n
        # print(act_dim, obs_dim)
        model = QMIX(
            training=True,
            agent_num=agent_num,
            obs_dim=obs_dim,
            act_num=act_dim,
            hidden_layers=[256, 256],
            lr=0.03,
            gamma=0.95,
            replay_size=10000,
            batch_size=64,
            epsilon_max=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.9999,
            update_after=200,
            update_online_every=1,
            update_target_every=200,
            seed=SEED,
        )
        rew_plot = []
        for episode in range(5000):
            obs_dict = env.reset(seed=SEED)
            # print(obs_dict)
            # obs_n = strkey2intkey(obs_dict, env.possible_agents)
            rew = 0
            obs_n = np.array([obs_dict['agent_0'], obs_dict['agent_1'], obs_dict['agent_2']])
            for step in range(25):
                act_n = model.react(obs_n)
                # act_dict = intkey2strkey(act_n, env.possible_agents)
                act_dict = {'agent_0': act_n[0][0],
                            'agent_1': act_n[1][0],
                            'agent_2': act_n[2][0]}
                next_obs_dict, rews_dict, term_dict, trun_dict, _ = env.step(act_dict)

                next_obs_n = np.array([next_obs_dict['agent_0'], next_obs_dict['agent_1'], obs_dict['agent_2']])
                rews_n = np.array(list(rews_dict.values()), dtype=np.float32)
                terminated = all(term_dict.values())
                truncated = any(trun_dict.values())

                model.store(obs_n, act_n, next_obs_n, rews_n, terminated)

                loss_Q = model.train()
                rew += rews_n.mean()
                if terminated or truncated:
                    break
                else:
                    obs_dict = next_obs_dict
                    obs_n = next_obs_n

            print(f'Episode {episode} finished after {step+1} steps, episode_reward={rew}')
            rew_plot.append(rew)
        # x = [i for i in range(len(rew_plot))]
        plt.plot(rew_plot)
        plt.show()
        env.close()
