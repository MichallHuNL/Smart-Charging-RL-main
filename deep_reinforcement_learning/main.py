import torch as th

from smart_grid_environment.env.smart_grid_environment import SmartChargingEnv
from smart_grid_environment.utils.default_params import default_params
from smart_grid_environment.experiment.independent_actor_critic_experiment import ActorCriticExperiment

if __name__ == '__main__':

    # Executing this code-block defines a new experiment
    params = default_params()
    params['max_episode_length'] = 500
    params['max_steps'] = int(2E6)
    params['double_q'] = True
    num_agents = params.get('n_agents', 4)
    env = SmartChargingEnv(num_ports=num_agents, action_space_size=params.get('n_actions', 3))
    n_actions, state_dim = params.get('n_actions', 10), env.observation_space(env.agents[0]).shape[0]
    # The model has n_action policy heads and one value head

    # Full observability of the agents
    input_dim = num_agents * state_dim
    models = [th.nn.Sequential(th.nn.Linear(input_dim , 128), th.nn.ReLU(),
                               th.nn.Linear(128, 512), th.nn.ReLU(),
                               th.nn.Linear(512, 128), th.nn.ReLU(),
                               th.nn.Linear(128, n_actions + 1)) for _ in range(num_agents)]
    experiment = ActorCriticExperiment(params, models, env)

    # Re-executing this code-block picks up the experiment where you left off
    try:
        experiment.run()
    except KeyboardInterrupt:
        experiment.close()
    # experiment.plot_training()
