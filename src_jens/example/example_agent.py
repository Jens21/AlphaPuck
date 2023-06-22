import laserhockey.hockey_env as h_env

class example_agent():
    agent = h_env.BasicOpponent(weak=True)

    def act(self, obs, evaluation=False):
        return self.agent.act(obs)

    def push_observation(self, obs_agent_1, a1, next_obs_agent_1, obs_agent_2, a2, next_obs_agent_2):
        pass

    def train(self, frame_idx, writer):
        pass