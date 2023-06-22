import laserhockey.hockey_env as h_env
from torch.utils.tensorboard import SummaryWriter

from agent import dqn_agent

import numpy as np
import torch
np.random.seed(12345)
torch.manual_seed(54321)

EPISODES = 1_000
MAX_STEPS = 5_000

player1 = dqn_agent() # change to your agent
player2 = h_env.BasicOpponent(weak=True)

if __name__ == '__main__':
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
    writer = SummaryWriter()

    frame_idx = 0
    for episode_idx in range(EPISODES):
        env.seed(seed=np.random.randint(0, 1e8))
        obs_agent_1, _ = env.reset()
        obs_agent_2 = env.obs_agent_two()

        for step_idx in range(MAX_STEPS):
            a1, a1_idx = player1.act(obs_agent_1, frame_idx, evaluation=False)
            a2 = player2.act(obs_agent_2)
            next_obs_agent_1, reward, done, _, info = env.step(np.hstack([a1,a2]))
            reward -= info["reward_closeness_to_puck"]
            next_obs_agent_2 = env.obs_agent_two()

            frame_idx += 1

            player1.push_observation(obs_agent_1, a1, a1_idx, next_obs_agent_1, obs_agent_2, a2, next_obs_agent_2, reward, done)

            obs_agent_1, obs_agent_2 = np.copy(next_obs_agent_1), np.copy(next_obs_agent_2)
            player1.train(frame_idx, writer)

            if done or frame_idx == MAX_STEPS-1:
                break

    env.close()
    writer.close()