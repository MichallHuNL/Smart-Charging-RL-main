import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box
import torch

class SmartChargingEnv(ParallelEnv):
    metadata = {"render.modes": ["human"], "name": "neighborhood_charging_env"}

    # Define constants for clearer code
    ETA = float(0.9)  # charging efficiency
    P_MAX = float(0.5)  # maximum charging power of car
    DELTA_T = float(1)  # 1 hour
    B_MAX = float(40)  # in kWh, maximum battery capacity
    power_cost_constant = 0.5  # Constant for linear multiplication for cost of power
    charging_reward_constant = 5  # Constant for linear multiplication for charging reward
    non_full_ev_cost_constant = 20  # Cost for EV leaving without full charge
    over_peak_load_constant = 5  # Cost for going over peak load that is multiplied by load
    peak_load = 1.5  # Maximum allowed load
    rng = np.random.default_rng(seed=42)  # random number generator for price vector
    PRICE_VEC = np.array(
        [62.04, 61.42, 58.14, 57.83, 58.30, 62.49, 71.58, 79.36, 86.02, 78.04, 66.51, 64.53, 47.55, 50.00,
         63.20, 71.17, 78.28, 89.40, 93.73, 87.19, 77.49, 71.62, 70.06, 66.39]) / 10
    schedule = np.array(
        [[0., 5., 4., 3., 2., 1., 0., 3., 2., 1., 0., 3., 2., 1., 0., 3., 2., 1., 0., 0., 4., 3., 2., 1.],
         [2., 1., 0., 0., 0., 1., 0., 0., 0., 0., 6., 5., 4., 3., 2., 1., 0., 0., 3., 2., 1., 0., 1., 0.],
         [7., 6., 5., 4., 3., 2., 1., 0., 0., 4., 3., 2., 1., 0., 0., 0., 1., 0., 0., 0., 2., 1., 0., 0.],
         [2., 1., 0., 0., 0., 0., 0., 0., 5., 4., 3., 2., 1., 0., 0., 0., 0., 2., 1., 0., 0., 7., 6.,
          5.]])  # A list of shape (num_agents, time) of the schedule of when cars come to the EV
    ends = np.array([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1.],
                     [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
    PERIODS = 24  # 24 hours


    # TODO: add peakload
    def __init__(self, num_ports=4, max_soc=1, max_time=24, max_price=10, penalty_factor=0.1, beta=0.01, test = False):
        super().__init__()

        # Number of charging ports
        self.num_ports = num_ports

        # Maximum SOC of an EV arriving
        self.max_soc = max_soc

        # Maximum amount of time a car can be at the charging port
        self.max_time = max_time

        # Maximum price on the grid (since otherwise cannot model it) / kWh
        self.max_price = max_price

        # Penalty factor for the car
        self.penalty_factor = penalty_factor

        # Car battery decay rate (considering this but not too sure)
        self.beta = beta

        # All the different ports defined according to the interface
        self.possible_agents = [f'port_{i}' for i in range(self.num_ports)]
        self.agents = self.possible_agents[:]

        # Define action and observation spaces for each agent
        self.action_spaces = {agent: Box(low=-self.P_MAX, high=self.P_MAX, shape=(1,), dtype=np.float32) for agent in self.possible_agents}
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
        electricity_price = self.PRICE_VEC[0]
        self.state = {}
        for idx, agent in enumerate(self.agents):
            if self.schedule[idx, self._elapsed_steps] > 0:
                self.state.update({agent: np.array([
                    0.2,
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
        return int(agent[5])


    # TODO: add cost for total action greater than max_allowed action
    def step(self, actions):
        rewards = {}
        dones = {}
        truncations = {}
        infos = {}

        total_reward = 0
        total_cost = 0

        self._elapsed_steps += 1

        for agent, action in actions.items():
            idx = self.get_index(agent)
            soc, remaining_time, price, has_ev = self.state[agent]
            if self.test:
                action_clipped = action
            else:
                action_clipped = action[0].item()
            if action_clipped < -soc:
                action_clipped = -soc
            elif action_clipped > 1-soc:
                action_clipped = 1 - soc

            if has_ev == 1:
                # Apply action to SoC
                soc += action_clipped  # Charging or discharging action
                # soc = np.clip(soc, 0, self.max_soc)

                # Calculate reward
                cost = price * action_clipped * self.power_cost_constant
                total_cost += cost
                reward = -cost

                reward += action_clipped * self.charging_reward_constant


                if self._elapsed_steps < len(self.PRICE_VEC) and self.ends[idx, self._elapsed_steps]:
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
                            soc, remaining_time, price, has_ev = 0.2, self.schedule[idx, self._elapsed_steps], self.PRICE_VEC[
                                self._elapsed_steps], 1
                    else:
                        soc, remaining_time, price, has_ev = 0, -1, self.PRICE_VEC[self._elapsed_steps], 0
                price = self.PRICE_VEC[self._elapsed_steps]

            if agent not in rewards:
                rewards[agent] = 0


            self.state[agent] = np.array([soc, remaining_time, price, has_ev], dtype=np.float32)
            infos[agent] = {}

        # Centralized reward adjustment for COMA
        for agent in self.agents:
            rewards[agent] += total_reward / len(self.agents)

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
            print(f'{agent}: SoC={soc_str}, Remaining Time={time_str}, Price={state[2]:.2f}, Has EV={state[3]}')

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
    actions = {f'port_{i}': env.action_spaces[f'port_{i}'].sample() for i in range(env.num_ports)}
    state, rewards, dones, truncations, infos = env.step(actions)
    print("State after step:", state)
    print("Rewards:", rewards)
    print("Dones:", dones)
    print("Truncations:", truncations)
    print("Infos:", infos)

