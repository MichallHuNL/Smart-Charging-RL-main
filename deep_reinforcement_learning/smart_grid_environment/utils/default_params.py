def default_params():
    """ These are the default parameters used int eh framework. """
    return {# Debugging outputs and plotting during training
            'plot_frequency': 10,             # plots a debug message avery n steps
            'plot_train_samples': True,       # whether the x-axis is env.steps (True) or episodes (False)
            'print_when_plot': True,          # prints debug message if True
            'print_dots': False,              # prints dots for every gradient update
            # Environment parameters
            'run_steps': 2048,                # samples whole episodes if run_steps <= 0
            'max_episode_length': 300,        # maximum number of steps per episode
            # Runner parameters
            'max_episodes': int(1E6),         # experiment stops after this many episodes
            'max_steps': int(2E6),            # experiment stops after this many steps
            'multi_runner': False,             # uses multiple runners if True
            'parallel_environments': 4,       # number of parallel runners  (only if multi_runner==True)
            # Exploration parameters
            'epsilon_anneal_time': int(2),    # exploration anneals epsilon over these many steps
            'epsilon_finish': 1E-5,           # annealing stops at (and keeps) this epsilon
                                              # epsilon_finish should be 0 for on-policy sampling,
                                              # but pytorch sometimes produced NaN gradients if probabilities get
                                              # too close to zero (because then ln(0)=-infty)
            'epsilon_start': 1,               # annealing starts at this epsilon
            # Optimization parameters
            'lr': 5E-4,                       # learning rate of optimizer
            'gamma': 0.99,                    # discount factor gamma
            'batch_size': 2048,               # number of transitions in a mini-batch
            'grad_norm_clip': 1,              # gradent clipping if grad norm is larger than this
            # Actor-critic parameters
            'value_loss_param': 0.1,          # governs the relative impact of the value relative to policy loss
            'advantage_bias': True,           # whether the advantages have the value as bias
            'advantage_bootstrap': True,      # whether advantages use bootstrapping (alternatively: returns)
            'offpolicy_iterations': 0,        # how many off-policy iterations are performed
            'value_targets': 'returns',       # either 'returns' or 'td' as regression targets of the value function
            # PPO parameters
            'ppo_clipping': True,             # whether we use the PPO loss
            'ppo_clip_eps': 0.1,              # the epsilon for the PPO loss
            # Image input parameters (not used in this exercise)
            'pixel_observations': False,      # use pixel observations (we will not use this feature here)
            'pixel_resolution': (78, 78),     # scale image to this resoluton
            'pixel_grayscale': True,          # convert image into grayscale
            'pixel_add_last_obs': True,       # stacks 2 observations
            'pixel_last_obs_delay': 3,        # delay between the two stacked observations
           }