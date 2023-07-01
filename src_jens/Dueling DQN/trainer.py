import torch.nn.functional as F
import torch as th

class Trainer():
    def __init__(self, gamma, batch_size):
        self.gamma = gamma
        self.batch_size = batch_size

    def train(self, replay_buffer, net_policy, net_target, optim):
        batch = replay_buffer.sample(self.batch_size)
        obs, action_idx, reward, done, next_obs = batch

        net_policy.train()
        net_target.eval()

        q_values = net_policy(obs)
        q_values = q_values[th.arange(self.batch_size), action_idx]

        with th.no_grad():
            next_q_values = net_target(next_obs).max(dim=1).values
            trg = reward + self.gamma * next_q_values * (1 - done)

        loss = F.smooth_l1_loss(q_values, trg)

        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss.item()