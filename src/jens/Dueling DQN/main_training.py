import laserhockey.hockey_env as h_env
from torch.utils.tensorboard import SummaryWriter
import torch as th

from action_selection import ActionSelection
from trainer import Trainer
from replay_buffer import ReplayBuffer
from network import Network

import numpy as np
import torch
np.random.seed(12345)
torch.manual_seed(54321)

MAX_FRAMES = 70_000_000

MAX_BUFFER_SIZE = 100_000
N_WARMUP = 10_000
GAMMA = 0.95
BATCH_SIZE = 128

EPS_MIN = 0.05
EPS_DECAY_DURATION = 1_000_000

USE_WEAK_OPPONENT = True

def evaluate_vs_best_player(net_policy, net_player, env):
    l_winners = []
    for episode in range(400):
        env.seed(seed=np.random.randint(0, 1e8))
        obs_1, _ = env.reset()
        obs_2 = env.obs_agent_two()

        while True:
            a1, a1_idx = action_selection.select_action(obs=obs_1, eps=10, net=net_policy, evaluation=True)
            a2, a2_idx = action_selection.select_action(obs=obs_2, eps=10, net=net_player, evaluation=True)

            for _ in range(4):
                obs_1, reward, done, _, info = env.step(np.hstack([a1, a2]))
                obs_2 = np.copy(env.obs_agent_two())
                if done:
                    break

            if done or frame_idx == MAX_FRAMES - 1:
                # writer.add_scalar('Winner', info['winner'], frame_idx)
                l_winners.append(info['winner'])
                break

    l_winners = np.array(l_winners)
    n_won = (l_winners == 1).sum()
    n_total = (l_winners != 0).sum()

    return n_won>1 and n_won/n_total>0.55

def evaluation_vs_basic_opponent(net_policy, env):
    global USE_WEAK_OPPONENT

    l_winners = []
    opponent = h_env.BasicOpponent(weak=USE_WEAK_OPPONENT)
    for episode in range(100):
        env.seed(seed=np.random.randint(0, 1e8))
        obs_1, _ = env.reset()
        obs_2 = env.obs_agent_two()

        while True:
            a1, a1_idx = action_selection.select_action(obs=obs_1, eps=10, net=net_policy, evaluation=True)

            for _ in range(4):
                a2 = opponent.act(obs_2)
                obs_1, reward, done, _, info = env.step(np.hstack([a1, a2]))
                obs_2 = np.copy(env.obs_agent_two())
                if done:
                    break

            if done or frame_idx == MAX_FRAMES-1:
                # writer.add_scalar('Winner', info['winner'], frame_idx)
                l_winners.append(info['winner'])
                break

    l_winners = np.array(l_winners)
    n_won = (l_winners==1).sum()
    n_lost = (l_winners==-1).sum()
    n_draw = (l_winners==0).sum()

    if n_won >= 50:
        USE_WEAK_OPPONENT = False

    print('\nWon: {} ({:.2f})%'.format(n_won, n_won/len(l_winners)), flush=True)
    print('Lost: {} ({:.2f})%'.format(n_lost, n_lost/len(l_winners)), flush=True)
    print('Draw: {} ({:.2f})%\n'.format(n_draw, n_draw/len(l_winners)), flush=True)

    return n_won, n_lost, n_draw

if __name__ == '__main__':
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
    writer = SummaryWriter()

    net_policy = Network()
    net_target = Network()
    net_player = Network()
    net_target.load_state_dict(net_policy.state_dict())
    optim = th.optim.Adam(net_policy.parameters(), lr=1e-4, eps=1e-3)

    replay_buffer = ReplayBuffer(max_capacity=MAX_BUFFER_SIZE)
    trainer = Trainer(gamma=GAMMA, batch_size=BATCH_SIZE)
    action_selection = ActionSelection()

    # opponent = h_env.BasicOpponent(weak=True)

    frame_idx = 0
    l_winners = []
    while frame_idx < MAX_FRAMES:
        env.seed(seed=np.random.randint(0, 1e8))
        obs_1, _ = env.reset()
        obs_2 = env.obs_agent_two()

        while True:
            eps = max(EPS_MIN, 1 - frame_idx/EPS_DECAY_DURATION)
            if frame_idx % 10 == 0:
                writer.add_scalar('epsilon', eps, frame_idx)

            a1, a1_idx = action_selection.select_action(obs=obs_1, eps=eps, net=net_policy, evaluation=False)
            a2, a2_idx = action_selection.select_action(obs=obs_2, eps=eps, net=net_policy, evaluation=False)

            for _ in range(1):
                # a2 = opponent.act(obs_2)
                next_obs_1, reward, done, _, info = env.step(np.hstack([a1, a2]))
                next_obs_2 = np.copy(env.obs_agent_two())
                if done:
                    break

            frame_idx += 1
            replay_buffer.push(obs_1, a1_idx, info["winner"], np.abs(info["winner"]), next_obs_1)
            replay_buffer.push(obs_2, a2_idx, -info["winner"], np.abs(info["winner"]), next_obs_2) # TODO, maybe don't do this
            obs_1, obs_2 = np.copy(next_obs_1), np.copy(next_obs_2)

            if frame_idx >= N_WARMUP and frame_idx % 16 == 0:
                loss = trainer.train(replay_buffer, net_policy, net_target, optim)
                writer.add_scalar('Loss', loss, frame_idx)

            if frame_idx % 50_000 == 0 and frame_idx > N_WARMUP:
                n_won, n_lost, n_draw = evaluation_vs_basic_opponent(net_policy, env)
                writer.add_scalar('Won', n_won, frame_idx)
                writer.add_scalar('Lost', n_lost, frame_idx)
                writer.add_scalar('Draw', n_draw, frame_idx)
                th.save(net_policy.state_dict(), 'net.pth')

            # if frame_idx % 20_000 == 0 and frame_idx > N_WARMUP:
            #     is_better = evaluate_vs_best_player(net_policy, net_player, env)
            #     if is_better:
            #         net_player.load_state_dict(net_policy.state_dict())
            #         print('update player')
            #     else:
            #         print("don't update player")

            if frame_idx % 2_000 == 0:
                net_target.load_state_dict(net_policy.state_dict())
                print('Frame: {}/{} ({:.2f}%)\tWinners: {}'.format(frame_idx, MAX_FRAMES, 100*frame_idx/MAX_FRAMES, np.mean(l_winners[-400:])), flush=True)

            if done or frame_idx == MAX_FRAMES-1:
                writer.add_scalar('Winner', info['winner'], frame_idx)
                l_winners.append(info['winner'])
                break

    env.close()
    writer.close()
