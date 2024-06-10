import numpy as np
import pylab as pl
from IPython import display
from matplotlib import pyplot as plt
import torch

from deep_reinforcement_learning.smart_grid_environment.env.smart_grid_environment import SmartChargingEnv
from deep_reinforcement_learning.smart_grid_environment.utils.plot import make_plots


class Experiment:
    """ Abstract class of an experiment. Contains logging and plotting functionality."""

    def __init__(self, params, models, **kwargs):
        self.params = params
        self.plot_frequency = params.get('plot_frequency', 100)
        self.plot_train_samples = params.get('plot_train_samples', True)
        self.print_when_plot = params.get('print_when_plot', False)
        self.print_dots = params.get('print_dots', False)
        self.episode_returns = []
        self.episode_lengths = []
        self.env_steps = []
        self.total_run_time = 0.0
        self.num_agents = params.get('n_agents', 4)
        self.agents = [f'port_{i}' for i in range(self.num_agents)]
        self.episode_losses = {agent: [] for agent in self.agents}
        self.models = models
        self.env = SmartChargingEnv(num_ports=params.get('n_agents', 4), test=True)
        self.plot()


    def plot(self):
        num_agents = 4
        n_steps = 24

        socs = np.zeros((n_steps, num_agents))
        actions = np.zeros((n_steps, num_agents))
        prices = np.zeros((n_steps))
        exists = np.zeros((n_steps, num_agents))
        remaining_times = np.zeros((n_steps, num_agents))

        obs = self.env.reset()[0]
        states = [obs[f'port_{i}'] for i in range(num_agents)]
        socs[0, :] = [state[0] for state in states]
        prices[0] = states[0][2]
        exists[0, :] = [state[3] for state in states]
        remaining_times[0, :] = [state[1] for state in states]

        for step in range(n_steps):
            action = {}
            for agent in range(num_agents):
                cur_state = torch.tensor(states[agent]).to(torch.float32)
                action_agent = self.models[agent](cur_state)[0]
                # action, _ = self.model.predict(obs)  # 1st step is based on reset()
                actions[step, agent] = action_agent
                action[f'port_{agent}'] = action_agent.detach()
            obs, reward, done, _, info = self.env.step(action)

            if step + 1 < n_steps:
                states = [obs[f'port_{i}'] for i in range(num_agents)]
                socs[step + 1, :] = [state[0] for state in states]
                prices[step + 1] = states[0][2]
                exists[step + 1, :] = [state[3] for state in states]
                remaining_times[step + 1, :] = [state[1] for state in states]
        print(actions)
        make_plots(socs, actions, prices, exists, remaining_times, np.transpose(np.array(self.env.ends)),
                   np.array(self.env.schedule))

    def plot_training(self, update=False):
        """ Plots logged training results. Use "update=True" if the plot is continuously updated
            or use "update=False" if this is the final call (otherwise there will be double plotting). """
        # Smooth curves
        window = max(int(len(self.episode_returns) / 50), 10)
        if any([len(self.episode_losses[agent]) < window + 2 for agent in self.agents]): return
        returns = np.convolve(self.episode_returns, np.ones(window) / window, 'valid')
        lengths = np.convolve(self.episode_lengths, np.ones(window) / window, 'valid')
        env_steps = np.convolve(self.env_steps, np.ones(window) / window, 'valid')
        for agent in self.agents:
            losses = np.convolve(self.episode_losses[agent], np.ones(window) / window, 'valid')
            # Determine x-axis based on samples or episodes
            if self.plot_train_samples:
                x_returns = env_steps
                x_losses = env_steps[(len(env_steps) - len(losses)):]
            else:
                x_returns = [i + window for i in range(len(returns))]
                x_losses = [i + len(returns) - len(losses) + window for i in range(len(losses))]
            # Create plot
            colors = ['b', 'g', 'r']
            fig = plt.gcf()
            fig.set_size_inches(16, 4)
            plt.clf()
            # Plot the losses in the left subplot
            pl.subplot(1, 3, 1)
            pl.plot(env_steps, returns, colors[0])
            pl.xlabel('environment steps' if self.plot_train_samples else 'episodes')
            pl.ylabel('episode return')
            # Plot the episode lengths in the middle subplot
            ax = pl.subplot(1, 3, 2)
            ax.plot(env_steps, lengths, colors[0])
            ax.set_xlabel('environment steps' if self.plot_train_samples else 'episodes')
            ax.set_ylabel('episode length')
            # Plot the losses in the right subplot
            ax = pl.subplot(1, 3, 3)
            ax.plot(x_losses, losses, colors[0])
            ax.set_xlabel('environment steps' if self.plot_train_samples else 'episodes')
            ax.set_ylabel('loss')

            plt.savefig(f'plots/{agent}.png')
            # dynamic plot update
            display.clear_output(wait=True)
            if update:
                display.display(pl.gcf())
        self.plot()

    def close(self):
        """ Frees all allocated runtime ressources, but allows to continue the experiment later.
            Calling the run() method after close must be able to pick up the experiment where it was. """
        pass

    def run(self):
        """ Starts (or continues) the experiment. """
        assert False, "You need to extend the Expeirment class and override the method run(). "