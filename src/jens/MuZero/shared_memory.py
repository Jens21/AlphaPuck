import ray

# shared memory acts as communication bridge between self play
# actors, trainer actors and the evaluator, which run in parallel, just shares the current network model
# Design of separating self play actors, replay buffer, shared memory and the train actor
# is derived from EfficientZero: https://arxiv.org/pdf/2111.00210.pdf but without the reanalyzing, context queue and batch queue
@ray.remote
class SharedMemory():
    current_model = None

    def __init__(self):
        pass

    def set_current_model(self, state_dict):
        self.current_model = state_dict

    def get_current_model(self):
        return self.current_model