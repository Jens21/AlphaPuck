import torch as th

class ReplayBuffer():
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity

        self.obs = th.empty((max_capacity, 18), dtype=th.float)
        self.action_idx = th.empty((max_capacity), dtype=th.long)
        self.reward = th.empty((max_capacity), dtype=th.float)
        self.done = th.empty((max_capacity), dtype=th.float)
        self.next_obs = th.empty((max_capacity, 18), dtype=th.float)

        self.buffer_size = 0
        self.idx = 0

    def __len__(self):
        return self.buffer_size

    def push(self, obs, action_idx, reward, done, next_obs):
        self.obs[self.idx] = th.from_numpy(obs).float()
        self.action_idx[self.idx] = th.LongTensor([action_idx])
        self.reward[self.idx] = th.FloatTensor([reward])
        self.done[self.idx] = th.FloatTensor([done])
        self.next_obs[self.idx] = th.from_numpy(next_obs).float()

        self.buffer_size = min(self.buffer_size+1, self.max_capacity)
        self.idx = (self.idx + 1) % self.max_capacity

    def sample(self, batch_size):
        indices = th.randint(0, self.buffer_size, (batch_size,))

        obs = self.obs[indices]
        action_idx = self.action_idx[indices]
        reward = self.reward[indices]
        done = self.done[indices]
        next_obs = self.next_obs[indices]

        return obs, action_idx, reward, done, next_obs