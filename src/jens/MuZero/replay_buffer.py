import torch as th
import ray

from utils import ACTIONS_T

# Design of separating self play actors, replay buffer, shared memory and the train actor
# is derived from EfficientZero: https://arxiv.org/pdf/2111.00210.pdf but without the reanalyzing, context queue and batch queue
@ray.remote
class ReplayBuffer():
    def __init__(self, max_buffer_size, K, unroll_steps, n_warmup, obs_dim=18):
        # Initialize replay buffer with a maximum capacity
        # max_buffer_size: Maximum number of samples

        self.do_train_step = False

        self.max_buffer_size = max_buffer_size
        self.K = K
        self.unroll_steps = unroll_steps
        self.n_warmup = n_warmup

        self.observations = th.empty((max_buffer_size, obs_dim), dtype=th.float)
        self.policy_distr = th.empty((max_buffer_size, ACTIONS_T.shape[0]), dtype=th.float)
        self.action_indices_1 = th.empty(max_buffer_size, dtype=th.long)
        self.action_indices_2 = th.empty(max_buffer_size, dtype=th.long)
        self.rewards = th.empty(max_buffer_size, dtype=th.float)
        self.dones = th.empty(max_buffer_size, dtype=th.float)

        self.indices = th.LongTensor([])
        self.priorities = th.FloatTensor([])
        self.idx = 0

        self.available_train_steps = 0

        self.total_frames = 0

    def get_len(self):
        return self.indices.shape[0]

    def set_priorities(self, indices, priorities):
        self.priorities[indices] = priorities

        self.do_train_step = False

    def get_do_train_step(self):
        return self.do_train_step

    def sample(self, batch_size):
        # Retrieve a batch of samples from the replay buffer
        # batch_size: Number of samples to retrieve
        if self.available_train_steps < 1:
            return None

        self.do_train_step = True

        self.available_train_steps -= 1

        # prios = self.priorities**self.alpha
        # prios = prios / prios.sum()
        # importance_weights = (1/self.priorities.shape[0] * 1/prios) ** self.beta
        # importance_weights /= importance_weights.max()

        # ind = th.multinomial(prios, batch_size, True)
        ind = th.randint(0, len(self.priorities), (batch_size,))

        # importance_weights = importance_weights[ind]

        # ind = th.randint(0, len(self.indices), (batch_size,))
        indices = self.indices[ind]

        expanded_indices = th.arange(self.unroll_steps + self.K + 1)[None, :] + indices[:, None] # [128, 3]
        expanded_indices = expanded_indices % self.max_buffer_size

        observations = self.observations[expanded_indices]
        action_indices_1 = self.action_indices_1[expanded_indices]
        action_indices_2 = self.action_indices_2[expanded_indices]
        rewards = self.rewards[expanded_indices]
        policy_distr = self.policy_distr[expanded_indices]
        dones = self.dones[expanded_indices]

        return observations, action_indices_1, action_indices_2, rewards, policy_distr, dones, None, None, self.total_frames


    def add_samples(self, observations, rewards, action_indices_1, action_indices_2, policy_distr, dones):
        # Add samples to the replay buffer
        # remember the last one contains no proper action_indices & rewards

        # observations: [N, 18]
        # rewards: [N]
        # action_indices: [N]

        n_samples = observations.shape[0]

        # 0, 1, ..., n_samples
        ind = th.arange(n_samples)
        ind_plus_mod = (ind + self.idx) % self.max_buffer_size

        # set the observations
        self.observations[ind_plus_mod] = observations
        self.action_indices_1[ind_plus_mod] = action_indices_1
        self.action_indices_2[ind_plus_mod] = action_indices_2
        self.rewards[ind_plus_mod] = rewards
        self.dones[ind_plus_mod] = dones
        self.policy_distr[ind_plus_mod] = policy_distr

        # if we added some indices that where at the beginning, remove them
        for i in ind_plus_mod:
            if self.indices.shape[0]!=0 and i==self.indices[0]:
                self.indices = self.indices[1:]
                self.priorities = self.priorities[1:]

        # add from which indices we can sample from
        self.indices = th.concat([self.indices, ind_plus_mod[:n_samples-self.K-self.unroll_steps]])
        max_priority = 1 if self.priorities.shape[0]==0 else self.priorities.max()
        self.priorities = th.concat([self.priorities, max_priority * th.ones(n_samples-self.K-self.unroll_steps)])
        self.idx = (self.idx + n_samples) % self.max_buffer_size

        # to make sure that every 4 sample steps there is only 1 training step
        if self.indices.shape[0] >= self.n_warmup:
            self.available_train_steps += (n_samples-self.K-self.unroll_steps) * (1./24.)
            self.total_frames += n_samples-self.K-self.unroll_steps
