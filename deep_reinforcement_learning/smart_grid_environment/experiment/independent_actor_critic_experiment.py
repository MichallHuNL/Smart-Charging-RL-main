from datetime import datetime

import numpy as np
from pettingzoo import ParallelEnv

from .experiment import Experiment
from ..controller.Controller import LogitsController, EpsilonGreedyController
from ..runner.Runner import Runner
from ..learner.qlearner import QLearner
from ..utils.TransitionBatch import TransitionBatch


class ActorCriticExperiment(Experiment):
    def __init__(self, params: dict, models, env: ParallelEnv, learner=None, **kwargs):
        super().__init__(params, models, **kwargs)
        self.max_episodes = params.get('max_episodes', int(1E6))
        self.max_steps = params.get('max_steps', int(1E9))
        self.grad_repeats = params.get('grad_repeats', 1)
        self.batch_size = params.get('batch_size', 1024)
        self.num_agents = params.get('n_agents', 4)
        self.agents = [f'port_{i}' for i in range(self.num_agents)]
        self.controller = LogitsController(models, num_actions=env.action_space(env.agents[0]).shape[0], params=params)
        self.controller = EpsilonGreedyController(controller=self.controller, params=params)
        self.runner = Runner(env, self.controller, params=params)
        self.learners = [QLearner(model, idx, params) if learner is None else learner for (idx, model) in
                         enumerate(models)]

    def close(self):
        """ Overrides Experiment.close() """
        self.runner.close()

    def run(self):
        """ Overrides Experiment.run() """
        # Plot past results if available
        if self.plot_frequency is not None and len(self.episode_losses) > 2:
            self.plot_training(update=True)
        # Run the experiment
        transition_buffer = {agent: TransitionBatch(self.batch_size, self.runner.transition_format(), self.batch_size)
                             for agent in self.agents}
        env_steps = 0 if len(self.env_steps) == 0 else self.env_steps[-1]
        for e in range(self.max_episodes):
            # Run the policy for batch_size steps
            batch = self.runner.run(self.batch_size, transition_buffer)
            env_steps += batch['env_steps']
            if batch['episode_length'] is not None:
                self.env_steps.append(env_steps)
                self.episode_lengths.append(batch['episode_length'])
                self.episode_returns.append(batch['episode_reward'])
            # Make a gradient update step
            for agent in self.agents:
                loss = [learner.train(batch['buffer'][agent]) for learner in self.learners]
                self.episode_losses[agent].append(np.mean(loss))
            # Quit if maximal number of environment steps is reached
            if env_steps >= self.max_steps: break
            # Show intermediate results
            if self.print_dots:
                print('.', end='')
            if self.plot_frequency is not None and (e + 1) % self.plot_frequency == 0 \
                    and len(self.episode_losses) > 2:
                self.plot_training(update=True)
                if self.print_when_plot:
                    print('Iteration %u, 100-epi-return %.4g +- %.3g, length %u, loss per agent %s' %
                          (len(self.episode_returns), np.mean(self.episode_returns[-100:]),
                           np.std(self.episode_returns[-100:]), np.mean(self.episode_lengths[-100:]),
                           [f'agent: {agent}, loss: {np.mean(self.episode_losses[agent][-100:])}' for agent in
                            self.agents]))
