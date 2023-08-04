import torch


class DuelingDQN(torch.nn.Module):

    def __init__(self, observation_dim, action_dim, lr=1e-3):
        super(DuelingDQN, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Replacement for the convolutional network to scale up to 512 nodes. Possibly add aditional layer with 256 nodes.
        self.conv_replacement = torch.nn.Sequential(
            torch.nn.Linear(self.observation_dim, 512),
            torch.nn.ReLU()
        )


        self.val_stream = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )

        self.adv_stream = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.action_dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        #self.double()


    def forward(self, state):

        input = self.conv_replacement(state)
        val = self.val_stream(input)
        adv = self.adv_stream(input)

        q = val + (adv - torch.mean(adv))

        return q