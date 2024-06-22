from copy import deepcopy

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


class QLearnerHardTarget(QLearner):
    def __init__(self, model, idx, params={}):
        super().__init__(model, idx, params)
        self.target_update = params.get('target_update', 'hard')
        self.target_update_interval = params.get('target_update_interval', 200)
        self.target_update_calls = 0
        if params.get('target_model', True):
            self.target_model = deepcopy(model)
            for p in self.target_model.parameters():
                p.requires_grad = False
        assert self.target_model is None or self.target_update == 'soft' or self.target_update == 'copy', \
            'If a target model is specified, it needs to be updated using the "soft" or "copy" options.'

    def q_values(self, states, target=False):
        if target:
            return self.target_model(states)
        return super().q_values(states)

    def target_model_update(self):
        if self.target_update == 'copy':
            self.target_update_calls += 1

            if self.target_update_calls >= self.target_update_interval:
                self.target_update_calls = 0
                self.target_model.load_state_dict(self.model.state_dict())
            return
        super().target_model_update()


class QLearnerSoftTarget(QLearnerHardTarget):
    def __init__(self, model, idx, params={}):
        super().__init__(model, idx, params)
        self.target_update = params.get('target_update', 'soft')
        self.soft_target_update_param = params.get('soft_target_update_param', 0.1)

    def target_model_update(self):
        if self.target_update == 'soft':
            self.target_update_calls += 1

            n = self.soft_target_update_param
            one_minus_n = 1 - n
            for name, param_target in self.target_model.named_parameters():
                param_model = self.model.get_parameter(name)
                param_target.data = one_minus_n * param_target + n * param_model
            return
        super().target_model_update()


class DoubleQLearner(QLearnerSoftTarget):
    def __init__(self, model, idx, params={}):
        super().__init__(model, idx, params)
        self.double_q = params.get('double_q', True)

    def _next_state_values(self, batch):
        """ Computes the Q-values of the 'states' and 'actions' of the given "batch". """
        if self.double_q:
            with th.no_grad():  # Next state values do not need gradients in DQN
                network_q = self.q_values(batch['next_states'][self.idx])
                q_indices = network_q.max(dim=-1, keepdim=True)[1]

                target_q = self.q_values(batch['next_states'][self.idx], target=True)
                return target_q.gather(dim=-1, index=q_indices)
        return super()._next_state_values(batch)