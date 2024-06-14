from datetime import datetime

import numpy as np
from pettingzoo import ParallelEnv

from .experiment import Experiment
from ..controller.Controller import LogitsController, EpsilonGreedyController
from ..runner.Runner import Runner
from ..learner.qlearner import QLearner, DoubleQLearner
from ..utils.TransitionBatch import TransitionBatch
from ..learner.comalearner import COMALearner


class ActorCriticExperiment(Experiment):
    def __init__(self, params: dict, models, env: ParallelEnv, critic=None, learner=None, **kwargs):
        super().__init__(params, models, **kwargs)
        self.max_episodes = params.get('max_episodes', int(1E6))
        self.max_steps = params.get('max_steps', int(1E9))
        self.grad_repeats = params.get('grad_repeats', 1)
        self.batch_size = params.get('batch_size', 1024)
        self.num_agents = params.get('n_agents', 4)
        self.method = params.get('method', 'IQL')
        self.agents = [i for i in range(self.num_agents)]
        self.controller = LogitsController(models, num_actions=env.n_actions, params=params)
        self.controller = EpsilonGreedyController(controller=self.controller, params=params)
        self.runner = Runner(env, self.controller, params=params)
        self.use_last_episode = params.get('use_last_episode', True)
        self.replay_buffer = [TransitionBatch(params.get('replay_buffer_size', int(1E5)),
                                             self.runner.transition_format(),
                                             batch_size=params.get('batch_size', 1024)) for agent in self.agents]
        if self.method == 'COMA':
            self.learner = COMALearner(models, critic, params)
        elif self.method == 'IQL':
            self.learners = [DoubleQLearner(model, idx, params) if learner is None else learner for (idx, model) in enumerate(models)]

    def _learn_from_episode(self, episode):
        """ This function uses the episode to train.
            Although not implemented, one could also add the episode to a replay buffer here.
            Returns the training loss for logging or None if train() was not called. """
        total_agent_loss = np.zeros(self.num_agents)
        for agent in self.agents:
            self.replay_buffer[agent].add(episode['buffer'][agent])
            if self.replay_buffer[agent].size >= self.replay_buffer[agent].batch_size:
                batch = self.replay_buffer[agent].sample()
                # Call train (params['grad_repeats']) times
                total_loss = 0
                for i in range(self.grad_repeats):
                    total_loss += self.learners[agent].train(batch)
                total_agent_loss[agent] = total_loss
        return total_agent_loss / self.grad_repeats

    def close(self):
        """ Overrides Experiment.close() """
        self.runner.close()

    def run(self):
        """ Overrides Experiment.run() """
        # Plot past results if available
        # if self.plot_frequency is not None and len(self.episode_losses) > 2:
        #     self.plot_training(update=True)
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
            if self.method == 'IQL':
                # for agent in self.agents:
                #     loss = [learner.train(batch['buffer'][agent]) for learner in self.learners]
                #     self.episode_losses[agent].append(np.mean(loss))
                losses = self._learn_from_episode(batch)
                for agent, loss in enumerate(losses):
                    self.episode_losses[agent].append(loss.item())
            elif self.method == 'COMA':
                # Make a gradient update step
                policy_losses, critic_loss = self.learner.train(batch['buffer'])
                for agent, policy_loss in zip(self.agents, policy_losses):
                    self.episode_losses[agent].append(policy_loss.item())
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

    def plot_training(self, update=False):
        window = max(int(len(self.episode_returns) / 50), 10)
        if any([len(self.episode_losses[agent]) < window + 2 for agent in self.agents]): return
        super().plot_training(update)
        # self.runner.plot()
