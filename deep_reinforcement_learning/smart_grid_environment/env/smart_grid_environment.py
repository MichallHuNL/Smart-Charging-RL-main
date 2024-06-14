import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete
import torch
import math

from deep_reinforcement_learning.smart_grid_environment.data.prices import EnergyPrices
from deep_reinforcement_learning.smart_grid_environment.utils.schedule import calculate_schedule


class SmartChargingEnv(ParallelEnv):
    metadata = {"render.modes": ["human"], "name": "neighborhood_charging_env"}

    power_cost_constant = 1  # Constant for linear multiplication for cost of power
    charging_reward_constant = 0.1  # Constant for linear multiplication for charging reward
    non_full_ev_cost_constant = 20  # Cost for EV leaving without full charge
    over_peak_load_constant = 5  # Cost for going over peak load that is multiplied by load

    rng = np.random.default_rng(seed=42)  # random number generator for price vector
    PERIODS = 24  # 24 hours

    # TODO: add peakload
    def __init__(self, num_ports=4, leaving_soc=0.8, max_soc=1, max_time=24, penalty_factor=0.1,
                 beta=0.01, action_space_size=10, p_grid_max=1.5, p_max=0.5, test=False):
        super().__init__()

        # A list of shape (num_agents, time) of the schedule of when cars come to the EV
        self.schedule = np.empty((num_ports, self.PERIODS))
        self.ends = np.empty((num_ports, self.PERIODS))
        self.PRICE_VEC = np.empty(self.PERIODS)
        self.peak_load = p_grid_max  # Maximum allowed load
        self.P_MAX = p_max  # maximum charging power of car in % of soc

        self.price_loader = EnergyPrices()

        # Number of charging ports
        self.num_ports = num_ports

        # Minimum soc to not receive punishment
        self.leaving_soc = leaving_soc - 0.01

        # Maximum SOC of an EV arriving
        self.max_soc = max_soc

        # Maximum amount of time a car can be at the charging port
        self.max_time = max_time

        # Maximum price on the grid (since otherwise cannot model it) / kWh
        self.max_price = self.price_loader.max_price()

        # Penalty factor for the car
        self.penalty_factor = penalty_factor

        # Car battery decay rate (considering this but not too sure)
        self.beta = beta

        self.n_actions = action_space_size

        # All the different ports defined according to the interface
        self.possible_agents = [i for i in range(self.num_ports)]
        self.agents = self.possible_agents[:]

        # Define action and observation spaces for each agent
        self.action_spaces = {agent: Discrete(self.n_actions, seed=self.rng) for agent in
                              self.possible_agents}
        self.observation_spaces = {
            agent: Box(
                low=np.array([0, -1, 0, 0]),
                high=np.array([self.max_soc, 24, self.max_price, 1]),
                dtype=np.float32
            ) for agent in self.possible_agents
        }
        self.test = test

        self._elapsed_steps = 0

        self.state = {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self._elapsed_steps = 0

        self.agents = self.possible_agents[:]

        # Get a price from the dataloader
        start_price = self.rng.integers(0, len(self.price_loader) - self.PERIODS)
        self.PRICE_VEC = self.price_loader[start_price:start_price + self.PERIODS] if options is None else options["prices"]
        # print(self.PRICE_VEC)

        # Set random arrival and departure times to train the agent on
        arrivals = self.rng.integers(self.PERIODS, size=self.PERIODS) if options is None else options["t_arr"]
        departures = self.rng.integers(arrivals, self.PERIODS + 1, size=self.PERIODS) if options is None else options["t_dep"]

        # Train the agent on days when cars don't leave
        # arrivals = np.zeros(self.num_agents, dtype=np.intp)
        # departures = np.ones(self.num_agents, dtype=np.intp) * (self.PERIODS - 1)

        self.schedule, self.ends = calculate_schedule(self.schedule.shape, arrivals, departures)

        electricity_price = self.PRICE_VEC[0]
        self.state = {}
        for idx, agent in enumerate(self.agents):
            if self.schedule[idx, self._elapsed_steps] > 0:
                self.state.update({agent: np.array([
                    self.rng.random() / 2 if options is None else options["soc_int"],
                    self.schedule[idx, self._elapsed_steps],
                    electricity_price,
                    1
                ])})
            else:
                self.state.update({agent: np.array([
                    0,
                    -1,
                    electricity_price,
                    0
                ])})

        return self.state, {agent: {} for agent in self.agents}


    def get_index(self, agent):
        return int(agent)


    # TODO: add cost for total action greater than max_allowed action
    def step(self, actions):
        rewards = {}
        dones = {}
        truncations = {}
        infos = {}
        total_action = 0

        total_reward = 0
        total_cost = 0

        self._elapsed_steps += 1

        for agent, action in actions.items():
            idx = self.get_index(agent)
            soc, remaining_time, price, has_ev = self.state[agent]

            if has_ev == 1:
                if not self.test:
                    action = action[0].item()

                action = ((float(action) / (self.n_actions // 2)) - 1.0) * self.P_MAX

                action_clipped = np.clip(action, -soc, self.max_soc - soc)

                total_action += action_clipped
                # Apply action to SoC
                soc += action_clipped  # Charging or discharging action

                # Calculate reward
                cost = price * action_clipped * self.power_cost_constant
                total_cost += cost
                reward = -cost

                reward += action_clipped * self.charging_reward_constant

                if soc < self.leaving_soc and self._elapsed_steps < len(self.PRICE_VEC) and self.ends[
                    idx, self._elapsed_steps]:
                    reward -= self.non_full_ev_cost_constant  # Penalty for car leaving without full charge

                rewards[agent] = reward
                total_reward += reward

                remaining_time -= 1

                if remaining_time <= 0:
                    has_ev = 0

            if self._elapsed_steps >= len(self.PRICE_VEC):
                dones[agent] = True
                truncations[agent] = False
                has_ev = 0  # Car leaves, port becomes empty
                soc = -1  # Undefined state for SoC
                remaining_time = -1  # Undefined state for remaining time

            else:
                dones[agent] = False
                truncations[agent] = False
                if has_ev < 1:
                    if self.schedule[idx, self._elapsed_steps] > 0:
                        soc, remaining_time, price, has_ev = 0.2, self.schedule[idx, self._elapsed_steps], \
                            self.PRICE_VEC[
                                self._elapsed_steps], 1
                    else:
                        soc, remaining_time, price, has_ev = 0, -1, self.PRICE_VEC[self._elapsed_steps], 0
                price = self.PRICE_VEC[self._elapsed_steps]

            if agent not in rewards:
                rewards[agent] = 0

            self.state[agent] = np.array([soc, remaining_time, price, has_ev], dtype=np.float32)
            infos[agent] = {}

        # Centralized reward adjustment for COMA
        total_reward -= self.over_peak_load_constant * np.max(total_action - self.peak_load, 0)

        for agent in self.agents:
            rewards[agent] += total_reward / self.num_ports

        all_done = all(dones.values())
        if all_done:
            self.agents = []
        else:
            self.agents = [agent for agent, done in dones.items() if not done]

        dones['__all__'] = all_done
        truncations['__all__'] = False

        return self.state, rewards, dones, truncations, infos

    def render(self):
        for agent, state in self.state.items():
            soc_str = f'{state[0]:.2f}' if state[0] >= 0 else 'N/A'
            time_str = f'{state[1]:.2f}' if state[1] >= 0 else 'N/A'
            print(f'agent {agent}: SoC={soc_str}, Remaining Time={time_str}, Price={state[2]:.2f}, Has EV={state[3]}')

    def close(self):
        pass

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def state(self):
        return np.array([self.state[agent] for agent in self.possible_agents])


# Example of creating the environment and running a step
if __name__ == '__main__':
    from pettingzoo.test import parallel_api_test

    env = SmartChargingEnv()
    state, _ = env.reset(seed=0)
    print('Check implementation: ', parallel_api_test(env))
    print("Initial state:", state)
    actions = {i: env.action_spaces[i].sample() for i in range(env.num_ports)}
    state, rewards, dones, truncations, infos = env.step(actions)
    print("State after step:", state)
    print("Rewards:", rewards)
    print("Dones:", dones)
    print("Truncations:", truncations)
    print("Infos:", infos)
