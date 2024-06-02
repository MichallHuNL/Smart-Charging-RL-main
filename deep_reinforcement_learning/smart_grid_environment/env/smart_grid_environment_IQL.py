import random
import numpy as np
from gymnasium.spaces import Box
import gymnasium
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from matplotlib import pyplot as plt
from deep_reinforcement_learning.smart_grid_environment.utils.plot import make_plots


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


class IQLSmartChargingEnv(gymnasium.Env):
    metadata = {"render.modes": ["human"], "name": "neighborhood_charging_env"}

    # Define constants for clearer code
    ETA = float(0.9)  # charging efficiency
    P_MAX = float(6.6)  # maximum charging power of car
    DELTA_T = float(1)  # 1 hour
    B_MAX = float(40)  # in kWh, maximum battery capacity
    rng = np.random.default_rng(seed=42)  # random number generator for price vector
    PRICE_VEC = np.array([62.04, 61.42, 58.14, 57.83, 58.30, 62.49, 71.58, 79.36, 86.02, 78.04, 66.51, 64.53, 47.55, 50.00,
                 63.20, 71.17, 78.28, 89.40, 93.73, 87.19, 77.49, 71.62, 70.06, 66.39]) / 10
    schedule = np.array([[0., 5., 4., 3., 2., 1., 0., 3., 2., 1., 0., 3., 2., 1., 0., 3., 2., 1., 0., 0., 4., 3., 2., 1.],
                [2., 1., 0., 0., 0., 1., 0., 0., 0., 0., 6., 5., 4., 3., 2., 1., 0., 0., 3., 2., 1., 0., 1., 0.],
                [7., 6., 5., 4., 3., 2., 1., 0., 0., 4., 3., 2., 1., 0., 0., 0., 1., 0., 0., 0., 2., 1., 0., 0.],
                [2., 1., 0., 0., 0., 0., 0., 0., 5., 4., 3., 2., 1., 0., 0., 0., 0., 2., 1., 0., 0., 7., 6., 5.]])  # A list of shape (num_agents, time) of the schedule of when cars come to the EV
    PERIODS = 24  # 24 hours

    def __init__(self, num_ports=4, max_soc=1, max_time=24, max_price=10, penalty_factor=0.1, beta=0.01, agent_nr = 0):
        super().__init__()

        self.agent_nr = agent_nr

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

        # Define action and observation spaces for each agent
        self.action_space = Box(low=-1, high=1, dtype=np.float32)

        low = []
        self.observation_space = Box(
            low=np.array([-1, -1, 0, 0]),
            high=np.array([self.max_soc, 24, self.max_price, 1]),
            dtype=np.float32
        )

        self.state = {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.t = 0
        electricity_price = self.PRICE_VEC[self.t]
        self.state = []

        if self.schedule[self.agent_nr, self.t] > 0.:
            self.state =  [0.2, self.schedule[self.agent_nr, self.t], electricity_price, 1]
        else:
            self.state = [-1, -1, electricity_price, 0]

        return np.array(self.state, dtype="float32"), {}


    # TODO: Needs to constraint that action cannot discharge or charge more than possible
    def step(self, action):
        reward = 0
        done = False
        truncation = False
        infos = {}

        self.t = self.t + 1

        soc, remaining_time, price, has_ev = self.state

        if has_ev == 1:
            # Apply action to SoC
            soc += action * self.P_MAX  # Charging or discharging action
            soc = np.clip(soc, 0, self.max_soc)[0]

            # Calculate reward
            cost = price * action

            reward = -cost

            if soc < 1 and remaining_time <= 0:
                reward -= 10  # Penalty for car leaving without full charge

            # Update remaining time
            remaining_time -= 1

            if remaining_time <= 0:
                has_ev = 0

        if self.t >= len(self.PRICE_VEC):
            done = True
            truncation = False
            has_ev = 0  # Car leaves, port becomes empty
            soc = -1  # Undefined state for SoC
            remaining_time = -1  # Undefined state for remaining time
        else:
            done = False
            truncation = False
            price = self.PRICE_VEC[self.t]
            if has_ev < 1:
                # print('here', self.schedule[idx_agent, self.t] )
                if self.schedule[self.agent_nr, self.t] > 0:
                    soc, remaining_time,  price, has_ev= 0.2, self.schedule[self.agent_nr, self.t], self.PRICE_VEC[self.t], 1
                else:
                    soc, remaining_time,  price, has_ev = -1, -1,  self.PRICE_VEC[self.t], 0




        self.state = [soc, remaining_time, price, has_ev]

        if action > 0.75:
            reward -= (2.25 + action) ** 2

        return np.array(self.state, dtype="float32"), reward, done, False, infos

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

    for i in range(num_agents):

        obs = envs[i].reset()[0]
        socs[0, i] = obs[0]
        prices[0] = obs[2]
        exists[0, i] = obs[3]
        remaining_times[0, i] = obs[1]

        for step in range(n_steps):
            action, _ = model.predict(obs)  # 1st step is based on reset()
            actions[step, i] = action
            obs, reward, done, _, info = envs[i].step(action)

            if step + 1 < n_steps:
                socs[step + 1, i] = obs[0]
                prices[step + 1] = obs[2]
                exists[step + 1, i] = obs[3]
                remaining_times[step + 1, i] = obs[1]


    make_plots(socs, actions, prices, exists, remaining_times)







if __name__ == '__main__':

    import os
    num_agents = 4
    envs = [IQLSmartChargingEnv(agent_nr=i) for i in range(num_agents)]
    state, _ = envs[0].reset(seed=0)
    print('Check implementation: ', check_env(envs[0]))
    print("Initial state:", state)
    actions = envs[0].action_space.sample()
    state, rewards, dones, truncations, infos = envs[0].step(actions)
    print("State after step:", state)
    print("Rewards:", rewards)
    print("Dones:", dones)
    print("Truncations:", truncations)
    print("Infos:", infos)

    # training
    n_timesteps = 10000  # 1 mil
    n_runs = 1  # 10 trial runs

    # instatiate path
    modeldir = f"PPO_model_{3}"
    logdir = f"PPO_log_{3}"

    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)


    # Force Stable Baselines3 to use CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


    # create model
    for j in range(num_agents):
        for i in range(n_runs):
            print(f"training run trial, {i + 1}")
            logname = f"Training_{i + 1}"
            model = PPO("MlpPolicy", env=envs[j], verbose=0, tensorboard_log=logdir, seed=i * 2)
            model.learn(total_timesteps=n_timesteps, progress_bar=True,
                        tb_log_name=logname)  # Train for a fixed number of timesteps

        # save model
        # model.save(f"{modeldir}/{logname}")

    get_info()
    # bagas1()
