import numpy as np
from matplotlib import pyplot as plt



def calculate_reward(soc, action, price, exists, remaining_time):
    P_MAX = float(6.6)
    if exists:
        soc += action * P_MAX  # Charging or discharging action
        soc = np.clip(soc, 0, 1)

        # Calculate reward
        cost = price * action
        reward = -cost

        if soc < 1 and remaining_time <= 0:
            reward -= 10  # Penalty for car leaving without full charge
        return reward
    else:
        return 0


def get_rewards(socs, actions, prices, exists, remaining_times):
    rewards = np.zeros((socs.shape))
    total_rewards = np.zeros((prices.shape))
    for i in range(len(prices)):
        for j in range(socs.shape[1]):
            reward = calculate_reward(socs[i, j], actions[i, j], prices[i], exists[i, j], remaining_times[i, j])
            rewards[i,j] = reward
            total_rewards[i] += reward
        if(sum(actions[i, :]) > 3):
            total_rewards -= sum(actions[i, :]) ** 2
    return rewards, total_rewards

# socs - numpy array of size (steps, num_agents)
# actions - numpy array of size (steps, num_agents)
# prices - numpy array of size (steps)
# exist - numpy array of size (steps, num_agents)
# remaining_times - numpy array of size (steps, num_agents)
def get_action_if_ev(actions, exists):
    action_if_ev = actions
    action_if_ev[exists != 1] = 0
    return action_if_ev



def make_plots(socs, actions, prices, exists, remaining_times):
    rewards, total_rewards = get_rewards(socs, actions, prices, exists, remaining_times)
    print(rewards)
    print(total_rewards)
    action_if_ev = get_action_if_ev(actions, exists)
    print(action_if_ev)

    for i in range(socs.shape[1]):
        title = f'Action, state of charge and prices for agent {i}'

        plt.subplot(2,2, i + 1)

        # plot soc_rl vs soc_exact
        plt.plot(np.arange(0, len(socs), 1), socs[:, i], label=f'soc-agent{1}')
        plt.bar(np.arange(0, len(actions), 1), actions[:, i], label=f'action-agent{i}')

        # set opacity
        plt.setp(plt.gca().patches, alpha=0.3)

        plt.xlabel("Time")
        plt.ylabel("Action")
        plt.title(title)
        plt.legend(loc="upper left")
        # create x tick labels
        plt.xticks(np.arange(0, len(socs), 1))

        # plot price in the same figure but on the right y-axis
        plt.twinx()
        plt.plot(prices * 10, label="Price", color='g')

        plt.ylabel("Price", color='g')
        plt.yticks(color='g')

        # display legend on top right
        plt.legend(loc="upper right")
        plt.tight_layout()
    plt.show()

    title = 'Total rewards'

    plt.plot(np.arange(0, len(socs), 1), total_rewards, label="total_rewards")

    # set opacity
    plt.setp(plt.gca().patches, alpha=0.3)

    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend(loc="upper left")
    # create x tick labels
    plt.xticks(np.arange(0, len(socs), 1))

    # plot price in the same figure but on the right y-axis
    plt.twinx()
    plt.plot(prices * 10, label="Price", color='g')

    plt.ylabel("Price", color='g')
    plt.yticks(color='g')

    # display legend on top right
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.show()