from matplotlib import pyplot as plt
import numpy as np
from mpe_env import mpe_env
from maddpg import MADDPG

SEED = 65535
ACTION_SPAN = 0.5


def run_mpe(save_file, actor_learning_rate, critic_learning_rate):
    env = mpe_env('simple_spread', seed=SEED)
    obs_dim, action_dim = env.get_space()
    agent_number = env.get_agent_number()
    # policy初始化
    maddpg_agents = MADDPG(
        training=True,
        obs_dim=obs_dim,
        act_num=action_dim,
        hidden_layers=[256, 256],
        lr=0.001,
        gamma=0.95,
        replay_size=10000,
        batch_size=32,
        start_steps=0,
        update_after=20,
        update_online_every=1,
        update_target_every=20,
        seed=SEED,
        tau=0.001,
        agent_num=agent_number,
        noise_range=0.1,
    )
    # 经验池初始化
    all_agent_exp = []

    score = []
    avg_score = []
    # 暖机，得到足够用来学习的经验
    obs_n = env.mpe_env.reset()
    terminated = False
    truncated = False

    for t in range(20):
        action_n = maddpg_agents.react(dict(obs_n))
        # action_n = [np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 0])]
        new_obs_n, reward_n, done_n, info_n = env.mpe_env.step(list(action_n.values()))
        maddpg_agents.store(dict(obs_n), action_n, dict(new_obs_n), dict(reward_n), terminated, truncated)
        obs_n = new_obs_n

    for i_episode in range(1000):
        obs_n = env.mpe_env.reset()
        score_one_episode = 0
        # 20是单个回合内的步数，可以设置的大一点
        for t in range(20):
            env.mpe_env.render()
            # 探索的幅度随机训练的进行逐渐减小
            action_n = maddpg_agents.react(dict(obs_n))
            # action_n = [np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 0])]
            new_obs_n, reward_n, done_n, info_n = env.mpe_env.step(list(action_n.values()))
            maddpg_agents.store(dict(obs_n), action_n, dict(new_obs_n), dict(reward_n), terminated, truncated)
            # 全都采用相同时刻的经验进行而学习
            maddpg_agents.train()
            score_one_episode += reward_n[0][0]
            obs_n = new_obs_n
        if (i_episode + 1) % 10 == 0:
            plt.plot(score)  # 绘制波形
            plt.plot(avg_score)  # 绘制波形

        score.append(score_one_episode)
        avg = np.mean(score[-100:])
        avg_score.append(avg)
        print(f"i_episode is {i_episode},score_one_episode is {score_one_episode},avg_score is {avg}")
    env.mpe_env.close()


if __name__ == '__main__':
    run_mpe('run64_3', 1e-2, 1e-2)
