import threading
import torch as th

import numpy as np


class MultiController:
    def __init__(self, models, num_actions=None, params={}):
        self.lock = threading.Lock()
        self.num_agents = params.get('num_agents', 1)
        self.num_actions = models[0][-1].out_features if num_actions is None else num_actions
        self.models = models

    def parameters(self):
        return self.model.parameters()

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
        probabilities = []
        for i in range(self.num_agents):
            try:
                mx = observations[i] if precomputed and precomputed[i] else self.models[i](
                    self.sanitize_inputs(observations[i]))
                probabilities.append(th.nn.functional.softmax(mx, dim=-1))
            finally:
                self.lock.release()
        return probabilities

    def choose(self, observation, **kwargs):
        return th.distributions.Categorical(probs=self.probabilities(observation, **kwargs)).sample()
