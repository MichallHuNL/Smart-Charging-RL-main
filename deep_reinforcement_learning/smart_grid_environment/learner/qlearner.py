import torch as th


class QLearner:
    """ A basic learner class that performs Q-learning train() steps. """

    def __init__(self, model, idx, params={}):
        self.model = model
        self.all_parameters = list(model.parameters())
        self.gamma = params.get('gamma', 0.99)
        self.optimizer = th.optim.Adam(self.all_parameters, lr=params.get('lr', 5E-4))
        self.criterion = th.nn.MSELoss()
        self.grad_norm_clip = params.get('grad_norm_clip', 10)
        self.target_model = None  # Target models are not yet implemented!
        self.idx = idx

    def target_model_update(self):
        """ This function updates the target network. No target network is implemented yet. """
        pass

    def q_values(self, states, target=False):
        """ Reutrns the Q-values of the given "states".
            I supposed to use the target network if "target=True", but this is not implemented here. """
        return self.model(states)

    def _current_values(self, batch):
        """ Computes the Q-values of the 'states' and 'actions' of the given "batch". """
        qvalues = self.q_values(batch['states'][self.idx])
        return qvalues.gather(dim=-1, index=batch['actions'][self.idx])

    def _next_state_values(self, batch):
        """ Computes the Q-values of the 'next_states' of the given "batch".
            Is greedy w.r.t. to current Q-network or target-network, depending on parameters. """
        with th.no_grad():  # Next state values do not need gradients in DQN
            # Compute the next states values (with target or current network)
            qvalues = self.q_values(batch['next_states'][self.idx], target=True)
            # Compute the maximum over Q-values
            return qvalues.max(dim=-1, keepdim=True)[0]

    def train(self, batch):
        """ Performs one gradient decent step of DQN. """
        self.model.train(True)
        # Compute TD-loss
        targets = batch['rewards'][self.idx] + self.gamma * (~batch['dones'][self.idx] * self._next_state_values(batch))
        loss = self.criterion(self._current_values(batch), targets.detach())
        # Backpropagate loss
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.all_parameters, self.grad_norm_clip)
        self.optimizer.step()
        # Update target network (if specified) and return loss
        self.target_model_update()
        return loss.item()
