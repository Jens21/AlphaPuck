import laserhockey.hockey_env as h_env
from torch.utils.tensorboard import SummaryWriter

from agents.dueling_jens import Dueling_Jens

import numpy as np
np.random.seed(12345)

ROUNDS = 100
RENDER = False

player1 = Dueling_Jens()
player2 = h_env.BasicOpponent(weak=False) # change to your agent
# player2 = Dueling_Jens()

if __name__ == '__main__':
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
    writer = SummaryWriter()

    l_winners = []
    for round_idx in range(ROUNDS):
        env.seed(seed=np.random.randint(0, 1e8))
        obs_agent_1, _ = env.reset()
        obs_agent_2 = env.obs_agent_two()

        while True:
            if RENDER: env.render()

            a1 = player1.act(obs_agent_1)
            a2 = player2.act(obs_agent_2)
            obs_agent_1, reward, done, _, info = env.step(np.hstack([a1,a2]))
            obs_agent_2 = env.obs_agent_two()

            if done:
                print('Winner: {}'.format('player 1' if info['winner'] == 1 else ('draw' if info['winner'] == 0 else 'player 2')))
                writer.add_scalar('Winner', info['winner'], round_idx)
                l_winners.append(info['winner'])
                break

    l_winners = np.array(l_winners)
    n_left_won = (l_winners == 1).sum()
    n_right_won = (l_winners == -1).sum()
    n_draws = (l_winners == 0).sum()

    print('\nLeft player won:\t{}/{}\t({:.2f}%)'.format(n_left_won, len(l_winners), 100*n_left_won/len(l_winners)))
    print('Right player won:\t{}/{}\t({:.2f}%)'.format(n_right_won, len(l_winners), 100*n_right_won/len(l_winners)))
    print('Draws:\t\t\t\t{}/{}\t({:.2f}%)'.format(n_draws, len(l_winners), 100*n_draws/len(l_winners)))

    env.close()
    writer.close()