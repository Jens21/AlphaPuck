import numpy as np
import torch as th
import ray

from own_env import OwnEnv
from action_selection import ActionSelection
from utils import ACTIONS_T
from network import Network

# Design of separating self play actors, replay buffer, shared memory and the train actor
# is derived from EfficientZero: https://arxiv.org/pdf/2111.00210.pdf but without the reanalyzing, context queue and batch queue
# self play agent
@ray.remote
class SelfPlay():
    def __init__(self, replay_buffer, gamma, K, unroll_steps, shared_memory, frame_skip):
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.K = K
        self.unroll_steps = unroll_steps
        self.shared_memory = shared_memory
        self.frame_skip = frame_skip

        self.action_selection = ActionSelection(self.gamma)

        self.env = OwnEnv()

        self.env.seed(seed=np.random.randint(0, 1e8))
        self.obs_1, _ = self.env.reset(customized=False)
        self.obs_2 = self.env.obs_agent_two()

        self.temp_obs_1 = []
        self.temp_action_idx_1 = []
        self.temp_action_idx_2 = []
        self.temp_reward = []
        self.policy_distr = []

        self.network = Network()
        self.network.load_state_dict(ray.get(self.shared_memory.get_current_model.remote()))

        self.step_idx = 0

    # just calls action selection, here some logging can be done
    def select_actions(self, net, obs, eps):
        action_1, action_idx_1, policy_distr = self.action_selection.select_action(eps, net, obs, False)

        return action_1, action_idx_1, policy_distr

     # stores the samples to a temporal buffer
    def store_sample(self, obs_1, action_idx_1, action_idx_2, reward, policy_distr):
        obs_1 = th.from_numpy(obs_1)

        self.temp_obs_1.append(obs_1)
        self.temp_action_idx_1.append(action_idx_1)
        self.temp_action_idx_2.append(action_idx_2)
        self.temp_reward.append(reward)
        self.policy_distr.append(policy_distr)

    # pushes the samples from the temporal buffer to the replay buffer
    def push_temp_buffer(self, last_obs_1, done):
        # if the game finished as done or truncated must be treated differently
        if done:
            n_often = self.K + self.unroll_steps

            default_policy = np.zeros(ACTIONS_T.shape[0])
            default_policy[-1] = 1.
            last_obs = th.from_numpy(last_obs_1).float()

            observations = th.from_numpy(np.vstack(self.temp_obs_1 + ((n_often+1)*[last_obs]))).float()
            rewards = th.FloatTensor([0] + self.temp_reward + (n_often*[0]))
            policy_distributions = th.from_numpy(np.vstack(self.policy_distr + ((n_often+1)*[default_policy]))).float()
            random_actions_1 = np.random.randint(0, ACTIONS_T.shape[0], n_often).tolist()
            action_indices_1 = th.LongTensor([-1] + self.temp_action_idx_1 + random_actions_1)
            random_actions_2 = np.random.randint(0, ACTIONS_T.shape[0], n_often).tolist()
            action_indices_2 = th.LongTensor([-1] + self.temp_action_idx_2 + random_actions_2)

            dones = th.zeros(action_indices_1.shape[0], dtype=th.float)
            dones[-(n_often+1):] = 1.
        else:
            observations = th.from_numpy(np.vstack(self.temp_obs_1)).float()
            rewards = th.FloatTensor([0] + self.temp_reward[:-1]).float()
            policy_distributions = th.from_numpy(np.vstack(self.policy_distr)).float()
            action_indices_1 = th.FloatTensor([0] + self.temp_action_idx_1[:-1]).long()
            action_indices_2 = th.FloatTensor([0] + self.temp_action_idx_2[:-1]).long()
            dones = th.zeros(observations.shape[0], dtype=th.float)

        # while ray.get(self.replay_buffer.get_do_train_step.remote()):
        #     time.sleep(0.1)

        self.replay_buffer.add_samples.remote(observations, rewards, action_indices_1, action_indices_2, policy_distributions, dones)
        self.temp_obs_1, self.temp_reward, self.temp_action_idx_1, self.temp_action_idx_2, self.policy_distr = [], [], [], [], []

    def get_last_lr(self):
        return self.scheduler.get_last_lr()[0]

    # the loop for the self play
    def do_self_play(self):
        while True:
            eps = 0.2
            action_1, action_idx_1, policy_distr = self.select_actions(self.network, self.obs_1, eps)
            action_2, action_idx_2, _ = self.action_selection.select_action(0.2, self.network, self.obs_2, False)
            self.step_idx += 1

            for _ in range(self.frame_skip):
                next_obs_1, reward, done, trunc, info = self.env.step(np.hstack([action_1, action_2]), customized=False)
                next_obs_2 = self.env.obs_agent_two()
                if trunc:
                    break

            # [obs, action_idx, reward, non_terminal_mask, next_obs]
            self.store_sample(self.obs_1, action_idx_1, action_idx_2, reward, policy_distr)

            self.obs_1 = np.copy(next_obs_1) # copying is probably not necessary
            self.obs_2 = np.copy(next_obs_2) # copying is probably not necessary

            if trunc:
                self.push_temp_buffer(next_obs_1, done)

                self.env.seed(seed=np.random.randint(0, 1e8))
                self.obs_1, _ = self.env.reset(customized=False)
                self.obs_2 = self.env.obs_agent_two()

                self.network.load_state_dict(ray.get(self.shared_memory.get_current_model.remote()))
