import numpy as np
import pylab as pl
from IPython import display
from matplotlib import pyplot as plt


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
        self.agents = [i for i in range(self.num_agents)]
        self.episode_losses = {agent: [] for agent in self.agents}

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

    def close(self):
        """ Frees all allocated runtime ressources, but allows to continue the experiment later.
            Calling the run() method after close must be able to pick up the experiment where it was. """
        pass

    def run(self):
        """ Starts (or continues) the experiment. """
        assert False, "You need to extend the Expeirment class and override the method run(). "