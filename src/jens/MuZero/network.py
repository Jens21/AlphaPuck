import torch as th
import torch.nn.functional as F

from utils import ACTIONS_T

# ResNetv2 preactivation block for the MLP case
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

# the main network
class Network(th.nn.Module):
    def __init__(self, obs_dim=18, action_dim=4):
        super(Network, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # tanh is important to prevent activation explosion (although only really necessary after the dynamics network)
        self.model_repres = th.nn.Sequential(
            th.nn.Linear(obs_dim, 64),
            ResidualBlock(inp_out_dim=64, hidden_dim=256),
            ResidualBlock(inp_out_dim=64, hidden_dim=256),
            th.nn.Tanh()
        )

        # Converts the latent state and both actions to the latent dim 64
        self.model_dynamic_linear = th.nn.Sequential(
            th.nn.Linear(64 + 2 * action_dim, 64)
        )
        # res bock 1 is used as common trunc for the dynamic model and the reward model later in the forward_dynamic function
        # tanh is important to prevent activation explosion
        self.model_dynamic_res_block = th.nn.Sequential(
            ResidualBlock(inp_out_dim=64, hidden_dim=256),
            ResidualBlock(inp_out_dim=64, hidden_dim=256),
            th.nn.Tanh()
        )

        # only has to predict loss, no winner, win for the next time step, hence shallow
        self.model_reward = th.nn.Sequential(
            th.nn.LayerNorm(64),
            th.nn.ReLU(),
            th.nn.Linear(64, 3),
        )

        # A shallow architecture seemed the best, however, using more residual blocks and longer training might succeed
        self.model_state_value = th.nn.Sequential(
            th.nn.Linear(64, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 1+ACTIONS_T.shape[0]),
            th.nn.Tanh()
        )

        # determined by looking in the code and empirical tests, tried to scale to [-1, 1]
        # some values are slightly larger than +-1 others like angular velocities can be larger than +-2 but usually never encounter in normal game play
        # time left of having puck is in [0, 1]
        self.observation_mean = th.FloatTensor([[-2.07, 0, 0, 0, 0, 0, 2.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.observation_std = th.FloatTensor([[1.57, 2.91, 1.04, 4, 4, 6, 1.57, 2.91, 1.04, 4, 4, 6, 3.7, 3, 12, 12, 15, 15]])

    def forward(self, x):
        raise NotImplementedError('This function should not be used!')

    # representation + prediction network (similar to MuZero pseudocode)
    def initial_inference(self, obs):
        obs_in = (obs - self.observation_mean) / self.observation_std

        latent_state = self.model_repres(obs_in)
        out = self.model_state_value(latent_state)
        state_values, policy_logits = out[..., 0], out[..., 1:]

        return latent_state, state_values, policy_logits

    # dynamics (and reward) + prediction network (similar to MuZero pseudocode)
    def recurrent_inference(self, latent_state, action_1, action_2):
        next_latent_state, rewards_logits = self.forward_dynamic(latent_state, action_1, action_2)

        out = self.model_state_value(next_latent_state)
        next_state_values, policy_logits = out[..., 0], out[..., 1:]

        # only these rewards can encounter, so only use them
        rewards = th.FloatTensor([-1, 0, 1])[rewards_logits.argmax(dim=-1)]

        return next_latent_state, rewards, rewards_logits, next_state_values, policy_logits

    def forward_dynamic(self, latent_state, action_1, action_2):
        net_inp = th.concat([latent_state, action_1, action_2], dim=1)
        state = self.model_dynamic_linear(net_inp)
        state = F.relu(state + latent_state)

        state = self.model_dynamic_res_block[0](state)
        reward_logits = self.model_reward(state)
        state = self.model_dynamic_res_block[1:](state)

        return state, reward_logits
