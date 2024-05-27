import gymnasium
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box
import gymnasium
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO


class SmartChargingEnv(gymnasium.Env):
   metadata = {"render.modes": ["human"], "name": "neighborhood_charging_env"}

   # Define constants for clearer code
   ETA = float(0.9)  # charging efficiency
   P_MAX = float(6.6)  # maximum charging power of car
   DELTA_T = float(1)  # 1 hour
   B_MAX = float(40)  # in kWh, maximum battery capacity
   rng = np.random.default_rng(seed=42)  # random number generator for price vector
   PRICE_VEC = rng.random(24) * 10  # random price vector for 24 hours multiplied by 10 (range 0-100)
   PERIODS = 24  # 24 hours

   def __init__(self, num_ports=4, max_soc=1, max_time=24, max_price=10, penalty_factor=0.1, beta=0.01):
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
       self.agent_index = {agent : idx for idx, agent in enumerate(self.possible_agents)}

       self.agents = self.possible_agents[:]

       # Define action and observation spaces for each agent
       self.action_space = Box(low=-1, high=1, shape=(self.num_ports,), dtype=np.float32)

       low = []
       self.observation_space = Box(
               low=np.array([-1, -1, 0, 0] * self.num_ports),
               high=np.array([self.max_soc, 24, self.max_price, 1] * self.num_ports),
               dtype=np.float32
           )

       self.state = {}

   def reset(self, seed=None, options=None):
       if seed is not None:
           np.random.seed(seed)

       self.agents = self.possible_agents[:]
       electricity_price = np.random.uniform(0, self.max_price)
       self.state = []

       for agent in self.agents:
           car_active = np.random.choice([0, 1])
           if car_active:
               self.state = self.state + [0.2, np.random.uniform(0, self.max_time), electricity_price, 1]
           else:
               self.state = self.state + [-1, -1, electricity_price, 0]

       return np.array(self.state, dtype="float32"), {agent: {} for agent in self.agents}

   def step(self, actions):
       rewards = {}
       dones = {}
       truncations = {}
       infos = {}

       total_reward = 0
       total_cost = 0

       for idx_agent, action in enumerate(actions):
           # print(idx_agent)
           # print(agent)
           soc, remaining_time, price, has_ev = self.state[idx_agent * 4 : (idx_agent + 1) * 4]
           agent = self.possible_agents[idx_agent]

           if has_ev == 1:
               # Apply action to SoC
               soc += action * self.P_MAX  # Charging or discharging action
               soc = np.clip(soc, 0, self.max_soc)

               # Calculate reward
               cost = price * abs(action)
               total_cost += cost
               reward = -cost

               if soc < 1 and remaining_time <= 0:
                   reward -= 10  # Penalty for car leaving without full charge

               rewards[agent] = reward
               total_reward += reward

               # Update remaining time
               remaining_time -= 1

               if remaining_time <= 0:
                   dones[agent] = True
                   truncations[agent] = False
                   has_ev = 0  # Car leaves, port becomes empty
                   soc = -1  # Undefined state for SoC
                   remaining_time = -1  # Undefined state for remaining time
               else:
                   dones[agent] = False
                   truncations[agent] = False

           else:
               rewards[agent] = 0
               dones[agent] = False
               truncations[agent] = False

           self.state[idx_agent * 4 : (idx_agent + 1) * 4] = [soc, remaining_time, price, has_ev]
           # self.state[agent * 4] = soc
           # self.state[agent * 4 + 1] = remaining_time
           # self.state[agent * 4 + 2] = price
           # self.state[agent * 4 + 3] = has_ev



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
if __name__ == '__main__':
    import os

    env = SmartChargingEnv()
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
    n_runs = 5  # 10 trial runs

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


