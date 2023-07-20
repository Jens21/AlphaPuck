import laserhockey.hockey_env as h_env
import torch as th

from action_selection import ActionSelection
from network import Network

import numpy as np
import torch
np.random.seed(12345)
torch.manual_seed(54321)

if __name__ == '__main__':
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)

    net = Network()
    net.load_state_dict(th.load('net.pth'))
    action_selection = ActionSelection()

    opponent = h_env.BasicOpponent(weak=True)

    frame_idx = 0
    l_winners = []
    for episode_idx in range(100):
        env.seed(seed=np.random.randint(0, 1e8))
        obs_1, _ = env.reset()
        obs_2 = env.obs_agent_two()

        while True:
            a1, a1_idx = action_selection.select_action(obs=obs_1, eps=10, net=net, evaluation=True)

            for _ in range(1):
                env.render('human')
                a2 = opponent.act(obs_2)
                next_obs_1, reward, done, _, info = env.step(np.hstack([a1, a2]))
                next_obs_2 = np.copy(env.obs_agent_two())
                if done:
                    print('Winner: {}'.format(info['winner']))
                    break

            frame_idx += 1
            obs_1, obs_2 = np.copy(next_obs_1), np.copy(next_obs_2)

            if done:
                l_winners.append(info['winner'])
                break

    l_winners = np.array(l_winners)
    n_won = (l_winners==1).sum()
    n_lost = (l_winners==-1).sum()
    n_draw = (l_winners==0).sum()

    print('\nWon: {} ({:.2f})%'.format(n_won, n_won/len(l_winners)))
    print('Lost: {} ({:.2f})%'.format(n_lost, n_lost/len(l_winners)))
    print('Draw: {} ({:.2f})%'.format(n_draw, n_draw/len(l_winners)))

    env.close()