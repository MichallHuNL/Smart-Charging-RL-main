import random
import numpy as np
from gymnasium.spaces import Box
import gymnasium
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from deep_reinforcement_learning.smart_grid_environment.utils.plot import make_plots, get_actions_clipped
from deep_reinforcement_learning.smart_grid_environment.utils.schedule import calculate_schedule


def create_car_schedule(number_cars, time):
    schedule = np.zeros((number_cars, time))
    for i in range(number_cars):
        for t in range(time):
            does_previous_car_exist = True if (t != 0 and schedule[i, t - 1] > 0) else False
            new_car = True if ((not does_previous_car_exist) and random.uniform(0, 1) > (2 / 3)) else False
            if does_previous_car_exist:
                schedule[i, t] = schedule[i, t - 1] - 1
            elif new_car:
                rand_hours_float = np.random.normal(scale=3, size=1) + 3
                rand_hours_int = max(np.round(rand_hours_float), 1)
                schedule[i, t] = rand_hours_int
    return schedule


class SingleSmartChargingEnv(gymnasium.Env):
    metadata = {"render.modes": ["human"], "name": "neighborhood_charging_env"}

    # Define constants for clearer code
    ETA = float(0.9)  # charging efficiency
    P_MAX = 0.5  # maximum charging power of car
    DELTA_T = float(1)  # 1 hour
    B_MAX = float(40)  # in kWh, maximum battery capacity
    power_cost_constant = 1 # Constant for linear multiplication for cost of power
    charging_reward_constant = 4 # Constant for linear multiplication for charging reward
    non_full_ev_cost_constant = 15 # Cost for EV leaving without full charge
    over_peak_load_constant = 5 # Cost for going over peak load that is multiplied by load
    peak_load = 0.9 # Maximum allowed load
    rng = np.random.default_rng(seed=42)  # random number generator for price vector
    PRICE_VEC = np.array([62.04, 61.42, 58.14, 57.83, 58.30, 62.49, 71.58, 79.36, 86.02, 78.04, 66.51, 64.53, 47.55, 50.00,
                 63.20, 71.17, 78.28, 89.40, 93.73, 87.19, 77.49, 71.62, 70.06, 66.39]) / 10
    schedule = np.array([[0., 5., 4., 3., 2., 1., 0., 3., 2., 1., 0., 3., 2., 1., 0., 3., 2., 1., 0., 0., 4., 3., 2., 1.],
                [2., 1., 0., 0., 0., 1., 0., 0., 0., 0., 6., 5., 4., 3., 2., 1., 0., 0., 3., 2., 1., 0., 1., 0.],
                [7., 6., 5., 4., 3., 2., 1., 0., 0., 4., 3., 2., 1., 0., 0., 0., 1., 0., 0., 0., 2., 1., 0., 0.],
                [2., 1., 0., 0., 0., 0., 0., 0., 5., 4., 3., 2., 1., 0., 0., 0., 0., 2., 1., 0., 0., 7., 6., 5.]])  # A list of shape (num_agents, time) of the schedule of when cars come to the EV
    ends = np.array([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
    PERIODS = 24  # 24 hours

    def __init__(self, num_ports=4, leaving_soc = 0.8, max_soc=1, max_time=24, max_price=10, penalty_factor=0.1, beta=0.01):
        super().__init__()

        # Number of charging ports
        self.num_ports = num_ports

        self.leaving_soc = leaving_soc

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
        self.possible_agents = [i for i in range(self.num_ports)]
        self.agent_index = {agent: idx for idx, agent in enumerate(self.possible_agents)}

        self.agents = self.possible_agents[:]

        # Define action and observation spaces for each agent
        self.action_space = Box(low=-1, high=1, shape=(self.num_ports,), dtype=np.float32)

        low = []
        self.observation_space = Box(
            low=np.array([0, -1, 0, 0] * self.num_ports),
            high=np.array([self.max_soc, 24, self.max_price, 1] * self.num_ports),
            dtype=np.float32
        )

        self.state = {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.t = 0
        self.agents = self.possible_agents[:]

        # TODO: get price based on real data
        self.PRICE_VEC = np.random.rand(*self.PRICE_VEC.shape) * self.max_price

        arrivals = self.rng.integers(self.PERIODS, size=self.PERIODS)
        departures = self.rng.integers(arrivals, self.PERIODS + 1, size=self.PERIODS)

        self.schedule, self.ends = calculate_schedule(self.schedule.shape, arrivals, departures)

        # print("schedule: ", self.schedule, flush=True)
        electricity_price = self.PRICE_VEC[self.t]
        self.state = []

        for idx, agent in enumerate(self.agents):
            if self.schedule[idx, self.t] > 0.:
                self.state = self.state +  [0.2, self.schedule[idx, self.t], electricity_price, 1]
            else:
                self.state = self.state + [0, -1, electricity_price, 0]

        return np.array(self.state, dtype="float32"), {agent: {} for agent in self.agents}


    # TODO: Needs to constraint that action cannot discharge or charge more than possible
    def step(self, actions):
        # rewards = {}
        dones = {}
        truncations = {}
        infos = {}

        total_reward = 0
        total_cost = 0

        total_action = 0

        self.t = self.t + 1

        for idx_agent, action in enumerate(actions):
            soc, remaining_time, price, has_ev = self.state[idx_agent * 4: (idx_agent + 1) * 4]
            agent = self.possible_agents[idx_agent]

            action_clipped = action * self.P_MAX
            if action_clipped < -soc:
                action_clipped = -soc
            elif action_clipped > 1-soc:
                action_clipped = 1 - soc

            if has_ev == 1:
                # Apply action to SoC
                soc += action_clipped  # Charging or discharging action

                # soc = np.clip(soc, 0, self.max_soc)

                # Calculate reward
                cost = price * action_clipped
                total_cost += cost
                total_reward -= cost * self.power_cost_constant

                total_reward += action_clipped * self.charging_reward_constant


                remaining_time -= 1

                if soc < self.leaving_soc and self.t < len(self.PRICE_VEC) and self.ends[idx_agent, self.t] == 1:
                    total_reward -= self.non_full_ev_cost_constant  # Penalty for car leaving without full charge

                total_action += action_clipped


                # Update remaining time

                if remaining_time <= 0:
                    has_ev = 0

            if self.t >= len(self.PRICE_VEC):
                dones[agent] = True
                truncations[agent] = False
                has_ev = 0  # Car leaves, port becomes empty
                soc = -1  # Undefined state for SoC
                remaining_time = -1  # Undefined state for remaining time
            else:
                dones[agent] = False
                truncations[agent] = False
                price = self.PRICE_VEC[self.t]
                if has_ev < 1:
                    # print('here', self.schedule[idx_agent, self.t] )
                    if self.schedule[idx_agent, self.t] > 0:
                        soc, remaining_time,  price, has_ev= 0.2, self.schedule[idx_agent, self.t], self.PRICE_VEC[self.t], 1
                    else:
                        soc, remaining_time,  price, has_ev = 0, -1,  self.PRICE_VEC[self.t], 0


                # else:
                    # print(idx_agent, self.t)



            self.state[idx_agent * 4: (idx_agent + 1) * 4] = [soc, remaining_time, price, has_ev]

            infos[agent] = {}

        if total_action > self.peak_load:
            total_reward -= total_action * self.over_peak_load_constant

        all_done = all(dones.values())
        if all_done:
            self.agents = []
        else:
            self.agents = [agent for agent, done in dones.items() if not done]

        dones['__all__'] = all_done
        truncations['__all__'] = False

        return np.array(self.state, dtype="float32"), total_reward, all_done, False, infos

    def render(self):
        for agent, state in self.state.items():
            soc_str = f'{state[0]:.2f}' if state[0] >= 0 else 'N/A'
            time_str = f'{state[1]:.2f}' if state[1] >= 0 else 'N/A'
            print(f'{agent}: SoC={soc_str}, Remaining Time={time_str}, Price={state[2]:.2f}, Has EV={state[3]}')

    def close(self):
        pass

    # def action_space(self, agent):
    #     return self.action_spaces[self.agent_index[agent]]
    #
    # def observation_space(self, agent):
    #     return self.observation_spaces[self.agent_index[agent]]

    def state(self):
        return self.state()


# Example of creating the environment and running a step

def get_info():
    num_agents = 4
    n_steps = 24

    socs = np.zeros((n_steps, num_agents))
    actions = np.zeros((n_steps, num_agents))
    prices = np.zeros((n_steps))
    exists = np.zeros((n_steps, num_agents))
    remaining_times = np.zeros((n_steps, num_agents))

    obs = env.reset()[0]
    socs[0, :] = [obs[i * 4] for i in range(num_agents)]
    prices[0] = obs[2]
    exists[0, :] = [obs[(i * 4) + 3] for i in range(num_agents)]
    remaining_times[0, :] = [obs[(i * 4) + 1] for i in range(num_agents)]

    for step in range(n_steps):
        action, _ = model.predict(obs)  # 1st step is based on reset()
        actions[step, :] = action
        obs, reward, done, _, info = env.step(action)

        if step + 1< n_steps:
            socs[step + 1, :] = [obs[i * 4] for i in range(num_agents)]
            prices[step + 1] = obs[2]
            exists[step + 1, :] = [obs[(i * 4) + 3] for i in range(num_agents)]
            remaining_times[step + 1, :] = [obs[(i * 4) + 1] for i in range(num_agents)]

    actions_clipped = get_actions_clipped(actions, socs, exists, env.P_MAX)


    make_plots(socs, actions_clipped, prices, exists, remaining_times, np.transpose(np.array(env.ends)), np.array(env.schedule))







if __name__ == '__main__':

    import os

    env = SingleSmartChargingEnv()
    state, _ = env.reset(seed=0)
    print('Check implementation: ', check_env(env))
    print("Initial state:", state)
    actions = env.action_space.sample()
    state, rewards, dones, truncations, infos = env.step(actions)
    print("State after step:", state)
    print("Rewards:", rewards)
    print("Dones:", dones)
    print("Truncations:", truncations)
    print("Infos:", infos)

    # training
    n_timesteps = 100000  # 1 mil
    n_runs = 1  # 10 trial runs

    # instatiate path
    modeldir = f"PPO_model_{2}"
    logdir = f"PPO_log_{2}"

    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)


    # Force Stable Baselines3 to use CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


    # create model
    for i in range(n_runs):
        print(f"training run trial, {i + 1}")
        logname = f"Training_{i + 1}"
        model = PPO("MlpPolicy", env=env, verbose=0, tensorboard_log=logdir, seed=i * 2)
        model.learn(total_timesteps=n_timesteps, progress_bar=True,
                    tb_log_name=logname)  # Train for a fixed number of timesteps

        # save model
        # model.save(f"{modeldir}/{logname}")

    get_info()
    # bagas1()
