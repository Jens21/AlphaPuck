import torch


class DuelingDQN(torch.nn.Module):

    def __init__(self, observation_dim, action_dim, lr=1e-3):
        super(DuelingDQN, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Replacement for the convolutional network
        self.conv_replacement = torch.nn.Sequential(
            torch.nn.Linear(self.observation_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU()
        )

        # Value stream of the dueling network
        self.val_stream = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

        # Advantage stream of the dueling network
        self.adv_stream = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.action_dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)



    def forward(self, state):

        input = self.conv_replacement(state)
        val = self.val_stream(input)
        adv = self.adv_stream(input)

        # combine the value and the advantage stream to receive the q value
        q = val + (adv - torch.mean(adv))

        return q
