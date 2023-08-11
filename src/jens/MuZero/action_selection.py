import numpy as np
import torch as th

from utils import ACTIONS_T

# class that selects the action for self-play and evaluation
# also includes epsilon strategy
class ActionSelection():
    def __init__(self, gamma):
        # Initialize the action selection algorithm
        # gamma: Discount factor for future rewards

        self.gamma = gamma

    # Chooses a random action for exploration
    def exploration(self):
        return np.random.randint(0, ACTIONS_T.shape[0])

    # this exploitation function is used during self-play
    # it plans only one time step into the future
    # also described in the report with the max min equation
    def exploitation_self_play(self, net, obs):
        net.eval()

        with th.no_grad():
            n_actions = ACTIONS_T.shape[0]

            # initial inference and broadcase all actions
            inp = th.from_numpy(obs).float()[None]
            latent_state_0, _, _ = net.initial_inference(inp)
            latent_state_0 = latent_state_0[None].tile(n_actions, n_actions, 1) # [7, 7, 32]
            action_1 = ACTIONS_T[None].tile(n_actions, 1, 1) # [7, 7, 32]
            action_2 = ACTIONS_T[:, None].tile(1, n_actions, 1) # [7, 7, 32]

            # reshaping of used variables and do recurrent inference
            latent_state_0, action_1, action_2 = latent_state_0.flatten(end_dim=1), action_1.flatten(end_dim=1), action_2.flatten(end_dim=1)
            _, rewards, _, next_state_values, _ = net.recurrent_inference(latent_state_0, action_1, action_2)
            rewards, next_state_values = rewards.reshape((n_actions, n_actions)), next_state_values.reshape((n_actions, n_actions))

            # computes the estimated state values for each action combination when planning one time step ahead
            state_val = rewards + self.gamma * next_state_values
            state_val = state_val.clip(-1, 1)
            state_val = state_val.min(dim=0).values
            max_action_idx = state_val.argmax(dim=0)

            # computes the target policy for the training, will be stored in the replay buffer
            # policy_distr = th.softmax(state_val, dim=0), also tried that version but it performs worse
            policy_distr = state_val + 1.01
            policy_distr /= policy_distr.sum()
            policy_distr = policy_distr**4
            policy_distr /= policy_distr.sum()

            return max_action_idx, policy_distr

    # this exploitation function plans multiple time steps in the future, only used for evaluation in the tournament since its slowly
    # could theoretically also be usd during self-play
    def exploitation_tournament(self, net, obs):
        # initilization of used variables
        obs_in = th.from_numpy(obs)[None].float()
        latent_states, state_values, policy_logits = net.initial_inference(obs_in)
        search_depths = th.FloatTensor([0])
        value_prefixes = th.FloatTensor([0])
        action_indices = th.arange(self.n_actions)

        # do the whole thing n_times, doesn't correspond exactly to search depth, but is often similar, especially for small values
        for i in range(self.n_times):
            # search the best found state values yet and investigate them further
            indices = th.argsort(state_values, descending=True)
            latent_states = latent_states[indices]  # [3, 64]
            state_values = state_values[indices]  # [3]
            search_depths = search_depths[indices]
            value_prefixes = value_prefixes[indices]
            if i != 0:
                action_indices = action_indices[indices]

            # plans one time step ahead with recurrent inference and some reshaping
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

            # calculates the estamated state values
            latent_out = latent_out.reshape(latent_best.shape)  # [n_best, n_actions, n_actions, obs_dim]
            rewards = rewards.reshape(latent_best.shape[:3])  # [n_best, n_actions, n_actions]
            next_state_values = next_state_values.reshape(latent_best.shape[:3])  # [n_best, n_actions, n_actions]
            value_prefixes_best = value_prefixes_best + rewards * (
                        self.gamma ** search_depths_best)  # [n_best, n_actions, n_actions]
            state_values_best = value_prefixes_best + next_state_values * (
                        self.gamma ** (search_depths_best + 1))  # [n_best, n_actions, n_actions]

            # computes the tensors that have to be appended to the previous ones
            n_mult = latent_out.shape[0] * latent_out.shape[1]
            state_values_append, min_indices = state_values_best.min(dim=2)  # [3, 25]
            latent_append = latent_out.flatten(end_dim=1)[th.arange(n_mult), min_indices.flatten()].reshape(
                (latent_out.shape[0], latent_out.shape[1], -1))
            value_prefixes_append = value_prefixes_best.flatten(end_dim=1)[
                th.arange(n_mult), min_indices.flatten()].reshape((latent_out.shape[0], latent_out.shape[1]))
            search_depths_append = search_depths[:self.n_best, None].tile(1, self.n_actions) + 1

            # append the newly calculated values
            state_values = th.concat([state_values[self.n_best:], state_values_append.flatten()])
            value_prefixes = th.concat([value_prefixes[self.n_best:], value_prefixes_append.flatten()])
            latent_states = th.concat([latent_states[self.n_best:], latent_append.flatten(end_dim=1)])
            search_depths = th.concat([search_depths[self.n_best:], search_depths_append.flatten()])

            # not necessary for i=0 since they are set before the for loop
            if i != 0:
                action_indices = th.concat(
                    [action_indices[self.n_best:], action_indices[:self.n_best, None].tile(1, self.n_actions).flatten()])

        # get the action with the best state value
        idx = th.argmax(state_values)
        max_action_idx = action_indices[idx].item()

        # computes the target policy for the training, will be stored in the replay buffer
        policy_distr = state_values + 1.01
        policy_distr /= policy_distr.sum()
        policy_distr = policy_distr ** 4
        policy_distr /= policy_distr.sum()

        return max_action_idx, policy_distr

    def select_action(self, eps, net, observation, evaluation=False, tournament=False):
        # Select an action based on the epsilon-greedy strategy
        # eps: The exploration factor
        # net: The neural network model
        # observation: The current observation
        # evaluation: Flag indicating whether it's for evaluation or not

        if tournament:
            action_idx_exploitation, policy_distr = self.exploitation_tournament(net, observation)
        else:
            action_idx_exploitation, policy_distr = self.exploitation_self_play(net, observation)

        # epsilon-greedy strategy
        if evaluation or np.random.uniform(0, 1) > eps:
            action_idx = action_idx_exploitation
        else:
            action_idx = self.exploration()

        return ACTIONS_T[action_idx], action_idx, policy_distr
