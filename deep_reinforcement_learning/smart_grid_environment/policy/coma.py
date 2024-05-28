from gymnasium.wrappers import Pol
from torch.optim import Adam


class COMA:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        self.actors_optimizer = Adam(lr=args.actors_lr)
        # We need a critic in COMA
        # We need policies for all individual agents
        # We need a target critic and an evaluation critic to remove the oscillation