import torch as th
import os

from smart_grid_environment.env.smart_grid_environment import SmartChargingEnv
from smart_grid_environment.utils.default_params import default_params
from smart_grid_environment.experiment.independent_actor_critic_experiment import ActorCriticExperiment
from tests.instance_loader import load_instance

if __name__ == '__main__':
    filename = "plots/plot.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    N, id = 5, 1
    num_agents, t_arr, t_dep, soc_req, soc_int, P_c_max, P_d_max, P_max_grid, E_cap, prices = (
        load_instance(N, id=id, filename='tests/test_instances.json'))
    assert P_c_max == P_d_max

    # Executing this code-block defines a new experiment
    params = default_params()
    params['n_actions'] = 10
    params['n_agents'] = num_agents
    params['max_steps'] = int(2E6)
    params['double_q'] = True

    # When these params change, retrain agents. Also retrain when num_agents changes.
    params['soc_req'] = soc_req[0]
    params['p_max'] = P_c_max[0] / E_cap[0]
    params['p_max_grid'] = P_max_grid[0] / E_cap[0]

    # Settings for plotting and checkpoints
    params['checkpoint_name'] = f"instance_N_{N}_id_{id}"
    params['e_cap'] = E_cap[0]

    env = SmartChargingEnv(num_ports=num_agents, action_space_size=params.get('n_actions'), p_max=params.get('p_max'),
                           p_grid_max=params.get('p_max_grid'), leaving_soc=params.get('soc_req'), )
    n_actions, state_dim = params.get('n_actions', 10), env.observation_space(env.agents[0]).shape[0]
    # The model has n_action policy heads and one value head

    # Full observability of the agents
    input_dim = num_agents * state_dim
    models = [th.nn.Sequential(th.nn.Linear(input_dim , 128), th.nn.ReLU(),
                               th.nn.Linear(128, 512), th.nn.ReLU(),
                               th.nn.Linear(512, 128), th.nn.ReLU(),
                               th.nn.Linear(128, n_actions + 1)) for _ in range(num_agents)]
    experiment = ActorCriticExperiment(params, models, env)

    experiment.load_checkpoint(230)
    experiment.test_instance(t_arr, t_dep, soc_int[0], prices)
    exit()

    # Re-executing this code-block picks up the experiment where you left off
    try:
        experiment.run()
    except KeyboardInterrupt:
        print("Stopped training")
    finally:
        # Now running test instance
        print("Running test instance")
        experiment.test_instance(t_arr, t_dep, soc_int[0], prices)
        experiment.close()
