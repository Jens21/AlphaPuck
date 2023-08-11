import numpy as np
import torch as th
import torch.nn.functional as F

device = 'cuda' if th.cuda.is_available() else 'cpu'

ACTIONS = [
    np.array([0, 0, 0, 1]),

    np.array([0, 1, 1, 0]),
    np.array([1, 1, 1, 0]),
    np.array([1, 0, 1, 0]),
    np.array([1, -1, 1, 0]),
    np.array([0, -1, 1, 0]),
    np.array([-1, -1, 1, 0]),
    np.array([-1, 0, 1, 0]),
    np.array([-1, 1, 1, 0]),

    np.array([0, 1, -1, 0]),
    np.array([1, 1, -1, 0]),
    np.array([1, 0, -1, 0]),
    np.array([1, -1, -1, 0]),
    np.array([0, -1, -1, 0]),
    np.array([-1, -1, -1, 0]),
    np.array([-1, 0, -1, 0]),
    np.array([-1, 1, -1, 0]),

    np.array([0, 1, 0, 0]),
    np.array([1, 1, 0, 0]),
    np.array([1, 0, 0, 0]),
    np.array([1, -1, 0, 0]),
    np.array([0, -1, 0, 0]),
    np.array([-1, -1, 0, 0]),
    np.array([-1, 0, 0, 0]),
    np.array([-1, 1, 0, 0]),
]

ACTIONS_T = th.from_numpy(np.array(ACTIONS)).to(device)

class ActionSelection():
    def __init__(self, gamma):
        # Initialize the action selection algorithm
        # gamma: Discount factor for future rewards

        self.gamma = gamma
        self.n_actions = ACTIONS_T.shape[0]
        self.n_times = 2
        self.n_best = 10

    def exploration(self):
        # Choose a random action for exploration

        return np.random.randint(0, ACTIONS_T.shape[0])

    def action_opponent(self, eps, net, obs):

        if np.random.uniform(0, 1) > eps:
            net.eval()
            with th.no_grad():
                inp = th.from_numpy(obs).float()[None]
                latent_state, state_values, policy_logits = net.initial_inference(inp)

                action_idx = policy_logits[0].argmax()

                return ACTIONS_T[action_idx], action_idx
        else:
            action_idx = self.exploration()

            return ACTIONS_T[action_idx], action_idx

    def exploitation(self, net, obs):
        net.eval()

        with th.no_grad():
            inp = th.from_numpy(obs).float()[None]
            latent_state_0, _, _ = net.initial_inference(inp)

            latent_state_0 = latent_state_0.tile((ACTIONS_T.shape[0], 1))

            latent_state_1, reward_1, _, _, _ = net.recurrent_inference(latent_state_0, ACTIONS_T)
            latent_state_1, reward_1 = latent_state_1[None].tile((ACTIONS_T.shape[0], 1, 1)), reward_1[None].tile((ACTIONS_T.shape[0], 1))
            actions = ACTIONS_T[:, None].tile((1, ACTIONS_T.shape[0], 1))
            latent_state_1, actions = latent_state_1.flatten(0, 1), actions.flatten(0, 1)

            latent_state_2, reward_2, _, state_values, _ = net.recurrent_inference(latent_state_1, actions)
            reward_2 = reward_2.reshape((ACTIONS_T.shape[0], ACTIONS_T.shape[0]))
            state_values = state_values.reshape((ACTIONS_T.shape[0], ACTIONS_T.shape[0]))
            total_state_values = reward_1 + self.gamma * reward_2 + (self.gamma ** 2) * state_values
            total_state_values = total_state_values.clip(-1, 1)

            max_action_idx = total_state_values.max(dim=0).values.argmax()

            policy_distr = th.softmax(total_state_values.max(dim=0).values, dim=0)

            return max_action_idx, policy_distr

    def exploitation_v2(self, net, obs):
        net.eval()

        with th.no_grad():
            n_actions = ACTIONS_T.shape[0]

            inp = th.from_numpy(obs).float()[None]
            latent_state_0, _, _ = net.initial_inference(inp)
            latent_state_0 = latent_state_0[None].tile(n_actions, n_actions, 1) # [7, 7, 32]
            action_1 = ACTIONS_T[None].tile(n_actions, 1, 1) # [7, 7, 32]
            action_2 = ACTIONS_T[:, None].tile(1, n_actions, 1) # [7, 7, 32]

            latent_state_0, action_1, action_2 = latent_state_0.flatten(end_dim=1), action_1.flatten(end_dim=1), action_2.flatten(end_dim=1)

            _, rewards, _, next_state_values, _ = net.recurrent_inference(latent_state_0, action_1, action_2)

            rewards, next_state_values = rewards.reshape((n_actions, n_actions)), next_state_values.reshape((n_actions, n_actions))

            state_val = rewards + self.gamma * next_state_values
            state_val = state_val.clip(-1, 1)
            state_val = state_val.min(dim=0).values
            max_action_idx = state_val.argmax(dim=0)

            # policy_distr = th.softmax(state_val, dim=0)
            policy_distr = state_val + 1.01
            policy_distr /= policy_distr.sum()
            policy_distr = policy_distr**4
            # policy_distr = th.zeros(ACTIONS_T.shape[0], dtype=th.float)
            # policy_distr[max_action_idx] = 1

            return max_action_idx, policy_distr

    def exploitation_v3(self, net, obs):
        obs_in = th.from_numpy(obs)[None].float().to(device)
        latent_states, state_values, policy_logits = net.initial_inference(obs_in)
        search_depths = th.FloatTensor([0]).to(device)
        value_prefixes = th.FloatTensor([0]).to(device)
        action_indices = th.arange(self.n_actions).to(device)

        for i in range(self.n_times):
            indices = th.argsort(state_values, descending=True)
            latent_states = latent_states[indices]  # [3, 64]
            state_values = state_values[indices]  # [3]
            search_depths = search_depths[indices]
            value_prefixes = value_prefixes[indices]
            if i != 0:
                action_indices = action_indices[indices]

            latent_best = latent_states[:self.n_best, None, None].tile(1, self.n_actions, self.n_actions,
                                                                  1)  # [n_best, n_actions, n_actions, obs_dim]
            actions1 = ACTIONS_T[None, :, None].tile(latent_best.shape[0], 1, self.n_actions,
                                                     1)  # [n_best, n_actions, n_actions, act_dim]
            actions2 = ACTIONS_T[None, None, :].tile(latent_best.shape[0], self.n_actions, 1,
                                                     1)  # [n_best, n_actions, n_actions, act_dim]
            search_depths_best = search_depths[:self.n_best, None, None].tile(1, self.n_actions,
                                                                         self.n_actions)  # [n_best, n_actions, n_actions]
            value_prefixes_best = value_prefixes[:self.n_best, None, None].tile(1, self.n_actions,
                                                                           self.n_actions)  # [n_best, n_actions, n_actions]

            latent_out, rewards, _, next_state_values, _ = net.recurrent_inference(latent_best.flatten(end_dim=2),
                                                                                   actions1.flatten(end_dim=2),
                                                                                   actions2.flatten(end_dim=2))

            latent_out = latent_out.reshape(latent_best.shape)  # [n_best, n_actions, n_actions, obs_dim]
            rewards = rewards.reshape(latent_best.shape[:3])  # [n_best, n_actions, n_actions]
            next_state_values = next_state_values.reshape(latent_best.shape[:3])  # [n_best, n_actions, n_actions]
            value_prefixes_best = value_prefixes_best + rewards * (
                        self.gamma ** search_depths_best)  # [n_best, n_actions, n_actions]
            state_values_best = value_prefixes_best + next_state_values * (
                        self.gamma ** (search_depths_best + 1))  # [n_best, n_actions, n_actions]

            n_mult = latent_out.shape[0] * latent_out.shape[1]
            state_values_append, min_indices = state_values_best.min(dim=2)  # [3, 25]
            latent_append = latent_out.flatten(end_dim=1)[th.arange(n_mult), min_indices.flatten()].reshape(
                (latent_out.shape[0], latent_out.shape[1], -1))
            value_prefixes_append = value_prefixes_best.flatten(end_dim=1)[
                th.arange(n_mult), min_indices.flatten()].reshape((latent_out.shape[0], latent_out.shape[1]))
            search_depths_append = search_depths[:self.n_best, None].tile(1, self.n_actions) + 1

            state_values = th.concat([state_values[self.n_best:], state_values_append.flatten()])
            value_prefixes = th.concat([value_prefixes[self.n_best:], value_prefixes_append.flatten()])
            latent_states = th.concat([latent_states[self.n_best:], latent_append.flatten(end_dim=1)])
            search_depths = th.concat([search_depths[self.n_best:], search_depths_append.flatten()])

            if i != 0:
                action_indices = th.concat(
                    [action_indices[self.n_best:], action_indices[:self.n_best, None].tile(1, self.n_actions).flatten()])

        idx = th.argmax(state_values)
        max_action_idx = action_indices[idx].item()

        policy_distr = state_values + 1.01
        policy_distr /= policy_distr.sum()
        policy_distr = policy_distr ** 4

        return max_action_idx, policy_distr

    def select_action(self, eps, net, observation, evaluation=False):
        # Select an action based on the epsilon-greedy strategy
        # eps: The exploration factor
        # net: The neural network model
        # observation: The current observation
        # evaluation: Flag indicating whether it's for evaluation or not

        # action_idx_exploitation, policy_distr = self.exploitation(net, observation)
        action_idx_exploitation, policy_distr = self.exploitation_v3(net, observation)

        if evaluation or np.random.uniform(0, 1) > eps:
            action_idx = action_idx_exploitation
        else:
            action_idx = self.exploration()

        return ACTIONS_T[action_idx], action_idx, policy_distr

class ResidualBlock(th.nn.Module):
    def __init__(self, inp_out_dim, hidden_dim=128):
        super(ResidualBlock, self).__init__()

        self.inp_out_dim = inp_out_dim

        self.model = th.nn.Sequential(
            th.nn.LayerNorm(inp_out_dim),
            th.nn.ReLU(),
            th.nn.Linear(inp_out_dim, hidden_dim),
            th.nn.LayerNorm(hidden_dim),
            th.nn.ReLU(),
            th.nn.Linear(hidden_dim, inp_out_dim)
        )

    def forward(self, x):
        out = self.model(x)

        return out + x


class Network(th.nn.Module):
    def __init__(self, obs_dim=18, action_dim=4):
        super(Network, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Dynamic network for predicting next observation and reward
        self.model_repres = th.nn.Sequential(
            th.nn.Linear(obs_dim, 64),
            # # th.nn.BatchNorm1d(128),
            # th.nn.ReLU(),
            # th.nn.Linear(128, 64),
            # th.nn.Tanh()
            ResidualBlock(inp_out_dim=64, hidden_dim=256),
            ResidualBlock(inp_out_dim=64, hidden_dim=256),
            th.nn.Tanh()
        ).to(device)

        self.model_dynamic_linear = th.nn.Sequential(
            th.nn.Linear(64 + 2 * action_dim, 64)
        ).to(device)
        self.model_dynamic_res_block = th.nn.Sequential(
            ResidualBlock(inp_out_dim=64, hidden_dim=256),
            ResidualBlock(inp_out_dim=64, hidden_dim=256),
            th.nn.Tanh()
        ).to(device)

        self.model_reward = th.nn.Sequential(
            # th.nn.Linear(64 + 2 * action_dim, 128),
            # th.nn.ReLU(),
            # th.nn.Linear(64, 3)

            th.nn.LayerNorm(64),
            th.nn.ReLU(),
            th.nn.Linear(64, 3),
        ).to(device)

        self.model_state_value = th.nn.Sequential(
            # ResidualBlock(inp_out_dim=64, hidden_dim=256),
            th.nn.LayerNorm(64),
            th.nn.ReLU(),
            th.nn.Linear(64, 256),
            th.nn.LayerNorm(256),
            th.nn.ReLU(),
            th.nn.Linear(256, 1+ACTIONS_T.shape[0]),
            th.nn.Tanh()
        ).to(device)

        self.observation_mean = th.FloatTensor([[-2.07, 0, 0, 0, 0, 0, 2.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).to(device)
        self.observation_std = th.FloatTensor([[1.57, 2.91, 1.04, 4, 4, 6, 1.57, 2.91, 1.04, 4, 4, 6, 3.7, 3, 12, 12, 15, 15]]).to(device)
        self.possible_rewards = th.FloatTensor([-1, 0, 1]).to(device)

    def forward(self, x):
        raise NotImplementedError('This function should not be used!')

    def initial_inference(self, obs):
        obs_in = (obs - self.observation_mean) / self.observation_std

        latent_state = self.model_repres(obs_in)
        out = self.model_state_value(latent_state)
        state_values, policy_logits = out[..., 0], out[..., 1:]

        return latent_state, state_values, policy_logits

    def recurrent_inference(self, latent_state, action_1, action_2):
        next_latent_state, rewards_logits = self.forward_dynamic(latent_state, action_1, action_2)

        out = self.model_state_value(next_latent_state)
        next_state_values, policy_logits = out[..., 0], out[..., 1:]

        # net_inp = th.concat([latent_state, action_1, action_2], dim=1)
        # rewards_logits = self.model_reward(net_inp)
        rewards = self.possible_rewards[rewards_logits.argmax(dim=-1)]

        return next_latent_state, rewards, rewards_logits, next_state_values, policy_logits

    def forward_dynamic(self, latent_state, action_1, action_2):
        net_inp = th.concat([latent_state, action_1, action_2], dim=1)
        state = self.model_dynamic_linear(net_inp)
        state = F.relu(state + latent_state)

        state = self.model_dynamic_res_block[0](state)
        reward_logits = self.model_reward(state)
        state = self.model_dynamic_res_block[1:](state)

        return state, reward_logits



class EfficientZero_Jens_Final_Single_Proc_2():
    def __init__(self, model_path) -> None:
        self.model = Network().eval()
        self.model.load_state_dict(th.load(model_path))

        self.action_selection = ActionSelection(0.975)

    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Acts in environment, yielding an action, given a state.
        S: State dimension.
        A: Action dimension.

        Args:
            state (np.ndarray): State [S].

        Returns:
            np.ndarray: Action (continuous) [A].
        """

        action, action_idx, policy_distr = self.action_selection.select_action(0., self.model, state, True)
        action = action.cpu().numpy()

        return action
