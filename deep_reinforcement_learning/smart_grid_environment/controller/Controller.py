import threading
import torch as th

import numpy as np


class MultiController:
    def __init__(self, models, num_actions=None, params={}):
        self.lock = threading.Lock()
        self.num_agents = params.get('n_agents', 4)
        self.num_actions = models[0][-1].out_features if num_actions is None else num_actions
        self.models = models

    def parameters(self):
        return [model.parameters() for model in self.models]

    def sanitize_inputs(self, observations):
        if isinstance(observations, np.ndarray) or isinstance(observations, list):
            observations = th.Tensor(observations).unsqueeze(dim=0)
        return observations

    def choose(self, observations):
        self.lock.acquire()
        actions = []
        for i in range(self.num_agents):
            try:
                mx = self.models[i](self.sanitize_inputs(observations[i]))
                if mx.shape[-1] > self.num_actions: mx = mx[:, :self.num_actions]
                actions.append(mx)
            finally:
                self.lock.release()
        return th.max(actions, dim=-1)[1]

    def probabilities(self, observations):
        self.lock.acquire()
        probabilities = []
        for i in range(self.num_agents):
            try:
                mx = self.models[i](self.sanitize_inputs(observations[i]))
                if mx.shape[-1] > self.num_actions: mx = mx[:, :self.num_actions]
                probabilities.append(mx)
            finally:
                self.lock.release()
        return [th.zeros(*mx.shape).scatter_(dim=-1, index=th.max(mx, dim=-1)[1].unsqueeze(dim=-1), src=th.ones(1, 1)) \
                for mx in probabilities]


class LogitsController(MultiController):
    """ A controller that interprets the first num_actions model outputs as logits of a softmax distribution. """

    def probabilities(self, observations, precomputed=None, **kwargs):
        self.lock.acquire()
        try:
            probabilities = th.zeros(self.num_agents, self.num_actions)
            for i in range(self.num_agents):
                port = i
                mx = observations[port] if precomputed and precomputed[i] else self.models[i](
                    self.sanitize_inputs(observations[port]))[:, :self.num_actions]
                probabilities[i] = th.nn.functional.softmax(mx, dim=-1)
                if th.isnan(probabilities).any():
                    print('Nan Found for following probabilities: ', probabilities)
                    print('Observations: ', observations)
                    print('MX: ', mx)
        finally:
            self.lock.release()
        return probabilities

    def choose(self, observation, **kwargs):
        return th.distributions.Categorical(probs=self.probabilities(observation, **kwargs)).sample()


class EpsilonGreedyController:
    """ A wrapper that makes any controller into an epsilon-greedy controller.
        Keeps track of training-steps to decay exploration automatically. """

    def __init__(self, controller, params={}, exploration_step=1):
        self.controller = controller
        self.num_actions = controller.num_actions
        self.max_eps = params.get('epsilon_start', 1.0)
        self.min_eps = params.get('epsilon_finish', 0.05)
        self.anneal_time = int(params.get('epsilon_anneal_time', 10000) / exploration_step)
        self.num_decisions = 0

    def epsilon(self):
        """ Returns current epsilon. """
        return max(1 - self.num_decisions / (self.anneal_time - 1), 0) \
            * (self.max_eps - self.min_eps) + self.min_eps

    def choose(self, observation, increase_counter=True, **kwargs):
        """ Returns the (possibly random) actions the agent takes when faced with "observation".
            Decays epsilon only when increase_counter=True". """
        eps = self.epsilon()
        if increase_counter: self.num_decisions += 1
        if np.random.rand() < eps:
            return th.randint(self.controller.num_actions, size=(self.controller.num_agents, ), dtype=th.long)
        else:
            return self.controller.choose(observation, **kwargs)

    def probabilities(self, observation, **kwargs):
        """ Returns the probabilities with which the agent would choose actions. """
        eps = self.epsilon()
        return eps * th.ones(1, 1) / self.num_actions + \
            (1 - eps) * self.controller.probabilities(observation, **kwargs)