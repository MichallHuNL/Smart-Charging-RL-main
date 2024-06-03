from abc import ABC, abstractmethod
import gymnasium
import torch as th
import numbers
import numpy as np
from pettingzoo import ParallelEnv

from ..utils.TransitionBatch import TransitionBatch


class Runner:

    def __init__(self, env: ParallelEnv, controller, params={}):
        self.env = env
        self.n_agents = params.get('n_agents', 4)
        self.agents = [i for i in range(self.n_agents)]
        self.cont_actions = isinstance(self.env.action_space(self.env.agents[0]), gymnasium.spaces.Box)
        self.controller = controller
        self.episode_length = params.get('max_episode_length', 300)
        self.gamma = params.get('gamma', 0.99)

        # Observations are equal to the state for the next state most likely
        self.state_shape = self.env.observation_space(self.env.agents[0]).shape
        self.sum_rewards = 0
        self.state = None
        self.time = 0
        self._next_step()

    def close(self):
        self.env.close()

    def transition_format(self):
        """ Returns the format of transitions: a dictionary of (shape, dtype) entries for each key. """
        return {'actions': ((1,), th.int64),
                'states': (self.state_shape, th.float32),
                'next_states': (self.state_shape, th.float32),
                'rewards': ((1,),  th.float32),
                'dones': ((1,), th.bool),
                'returns': ((1,), th.float32),
                'probabilities': ((self.env.n_actions,),  th.float32)
                }

    def _wrap_transition(self, s, a, r, ns, d, p):
        """ Takes a transition and returns a corresponding dictionary. """
        trans = {}
        form = self.transition_format()

        for key, val in [('states', s), ('actions', a), ('rewards', r), ('next_states', ns), ('dones', d), ('probabilities', p)]:
            if not isinstance(val, th.Tensor):
                if isinstance(val, numbers.Number) or isinstance(val, bool): val = [val]
                val = th.tensor(val, dtype=form[key][1])
            if len(val.shape) < len(form[key][0]) + 1: val = val.unsqueeze(dim=0)
            trans[key] = val
        return trans

    def _actions_to_dict(self, actions):
        d = {}
        for agent, action in enumerate(actions):
            d[agent] = [action]

        return d

    def _make_step(self, a):
        """ Make an actual step inside the environment given to the runner and retrieve information """
        ns, r, t, d, _ = self.env.step(a)
        self.sum_rewards += th.sum(th.tensor(list(r.values())), dim=0)
        return r, ns, t, d or t

    def _next_step(self, done=True, next_state=None):
        """Switch to the next step for the runner"""
        self.time = 0 if done else self.time + 1
        if done:
            self.sum_rewards = 0
            self.state, _ = self.env.reset()
        else:
            self.state = next_state

    def run(self, n_steps, transition_buffer=None, trim=True, return_dict=None):
        """ Run in the environment for n_steps and store them inside a transition_buffer"""
        max_steps = n_steps if n_steps > 0 else self.episode_length
        tb = {}
        for agent in self.agents:
            tb[agent] = TransitionBatch(max_steps, self.transition_format())
        time, episode_start, episode_lengths, episode_rewards = 0, 0, [], []
        for t in range(max_steps):
            action = self._actions_to_dict(self.controller.choose(self.state))
            r, ns, terminal, done = self._make_step(action)
            probs = self.controller.probabilities(self.state)
            done = done["__all__"]
            for agent in self.agents:
                tb[agent].add(self._wrap_transition(self.state[agent], action[agent], r[agent], ns[agent], terminal[agent], probs[agent]))
                # Terminate the Runner when there are no more runs allowed
                if self.env._elapsed_steps >= self.episode_length: done = True
                if done or t == (max_steps - 1):
                    tb[agent]['returns'][t] = tb[agent]['rewards'][t]
                    for i in range(t - 1, episode_start -1, -1):
                        # Bellman equation
                        tb[agent]['returns'][i] = tb[agent]['rewards'][i] + self.gamma * tb[agent]['returns'][i + 1]
                    episode_start = t + 1
                if done:
                    episode_lengths.append(self.time + 1)
                    episode_rewards.append(self.sum_rewards)
            self._next_step(done=done, next_state=ns)
            time += 1
            if done and n_steps <= 0:
                for agent in self.agents:
                    tb[agent].trim()
                break
            transition_buffer = {} if transition_buffer is None else transition_buffer
            for agent in self.agents:
                transition_buffer[agent] = tb[agent] if (agent not in transition_buffer) else transition_buffer[agent].add(tb[agent])
                if trim: transition_buffer[agent].trim()
        if return_dict is None: return_dict = {}
        return_dict.update({'buffer': transition_buffer,
                            'episode_reward': None if len(episode_rewards) == 0 else np.mean(episode_rewards),
                            'episode_length': None if len(episode_lengths) == 0 else np.mean(episode_lengths),
                            'env_steps': time})
        return return_dict

    def run_episode(self, transition_buffer=None, trim=True, return_dict=None):
        return self.run(0, transition_buffer, trim, return_dict)