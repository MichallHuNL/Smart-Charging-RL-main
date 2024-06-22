def default_params():
    """ These are the default parameters used int eh framework. """
    return {  # Debugging outputs and plotting during training
            'plot_frequency': 10,             # plots a debug message avery n steps
            'plot_train_samples': True,       # whether the x-axis is env.steps (True) or episodes (False)
            'print_when_plot': True,          # prints debug message if True
            'print_dots': False,              # prints dots for every gradient update
            # Environment parameters
            'max_episode_length': 25,         # maximum number of steps per episode


            # Runner parameters
            'max_episodes': int(1E6),         # experiment stops after this many episodes
            'max_steps': int(2E6),            # experiment stops after this many steps
            # Exploration parameters
            'epsilon_anneal_time': int(2),    # exploration anneals epsilon over these many steps
            'epsilon_finish': 1E-5,           # annealing stops at (and keeps) this epsilon
                                              # epsilon_finish should be 0 for on-policy sampling,
                                              # but pytorch sometimes produced NaN gradients if probabilities get
                                              # too close to zero (because then ln(0)=-infty)
            'epsilon_start': 1,               # annealing starts at this epsilon
            # Optimization parameters
            'lr': 5E-4,                       # learning rate of optimizer
            'critic_lr': 5E-4,                # learning rate of the critic
            'gamma': 0.99,                    # discount factor gamma
            'batch_size': 2048,               # number of transitions in a mini-batch
            'grad_norm_clip': 1,              # gradent clipping if grad norm is larger than this
            'n_actions': 10,                  # Size of discrete action space
            # DQN parameters
            'replay_buffer_size': int(1E4),   # the number of transitions in the replay buffer
            'use_last_episode': True,         # whether the last episode is always sampled from the buffer
            'target_model': True,             # whether a target model is used in DQN
            'target_update': 'soft',          # 'soft' target update or hard update by regular 'copy'
            'target_update_interval': 10,     # interval for the 'copy' target update
            'soft_target_update_param': 0.1,  # update parameter for the 'soft' target update
            'double_q': True,                 # whether DQN uses double Q-learning
            'grad_repeats': 4,                # how many gradient updates / runner call
           }
