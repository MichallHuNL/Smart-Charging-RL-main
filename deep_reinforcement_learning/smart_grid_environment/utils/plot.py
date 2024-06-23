import numpy as np
from matplotlib import pyplot as plt


power_cost_constant = 1
charging_reward_constant = 4
non_full_ev_cost_constant = 15
over_peak_load_constant = 5
peak_load = 1.5


def calculate_reward(soc, action, price, exists, end):
    if exists:
        # Cost of paying for electricity
        cost = price * action
        reward = -cost * power_cost_constant

        # Reward for charging the vehicle
        reward += action * charging_reward_constant

        if end == 1 and soc < 1:
            reward -= non_full_ev_cost_constant  # Penalty for car leaving without full charge
        return reward
    else:
        return 0


def get_rewards(socs, actions, prices, exists, remaining_times, ends):
    rewards = np.zeros((socs.shape))
    total_rewards = np.zeros((prices.shape))
    for i in range(len(prices)):
        for j in range(socs.shape[1]):
            reward = calculate_reward(socs[i, j], actions[i, j], prices[i], exists[i, j], ends[i, j])
            rewards[i,j] = reward
            total_rewards[i] += reward
        if(sum(actions[i, :]) > peak_load):
            total_rewards -= sum(actions[i, :]) * over_peak_load_constant
    return rewards, total_rewards


def get_action_if_ev(actions, exists):
    action_if_ev = actions
    action_if_ev[exists != 1] = 0
    return action_if_ev

def get_socs_when_leave(socs, actions, ends, p_max, soc_req):
    socs_plus_leaves = socs
    cars_left = 0.0
    correct = 0.0
    for i in range(socs.shape[0]):
        for j in range(socs.shape[1]):
            if ends[i, j] == 1:
                socs_plus_leaves[i, j] = np.clip(socs[i-1, j] + actions[i-1, j] * p_max, 0, 1)
                cars_left += 1
                correct += socs_plus_leaves[i, j] > soc_req
    return socs_plus_leaves, correct / cars_left


def find_non_zero_intervals(row):
    intervals = []
    start = None

    for i, val in enumerate(row):
        if val != 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            intervals.append((start, i - 1))
            start = None

    if start is not None:
        intervals.append((start, len(row) - 1))

    return intervals




# socs - numpy array of size (steps, num_agents)
# actions - numpy array of size (steps, num_agents)
# prices - numpy array of size (steps)
# exist - numpy array of size (steps, num_agents)
# remaining_times - numpy array of size (steps, num_agents)
# ends - numpy array of size (steps, num_agents)
# schedule - numpy array of size (steps, num_agents)
def make_plots(socs, pre_filter_actions, prices, exists, remaining_times, ends, schedule, rewards, p_max=0.2, E_cap=74, soc_req=0.8):
    actions = get_action_if_ev(pre_filter_actions, exists)
    # rewards, total_rewards = get_rewards(socs, actions, prices, exists, remaining_times, ends)
    socs, corrects = get_socs_when_leave(socs, actions, ends, p_max, soc_req)

    actions = np.clip(actions, -socs / p_max, (1 - socs) / p_max)

    total_cost = (np.transpose(prices) @ ((actions * p_max) * E_cap)).sum()
    print("total_cost", total_cost)
    print("% SoC correct", corrects)
    # print(action_if_ev)

    # Get intervals for each row
    intervals = [find_non_zero_intervals(row) for row in schedule]
    # print("intervals", intervals)




    for i in range(socs.shape[1]):
        title = f'Action, state of charge and prices for agent {i}'

        # plt.subplot(2,2, i + 1)
        fig1, ax1 = plt.subplots()

        # plot soc_rl vs soc_exact
        plt.plot(np.arange(0, len(socs), 1), socs[:, i], label=f'soc-agent{i}')

        # print("actions: ", actions[:, i], flush=True)
        plt.bar(np.arange(0, len(actions), 1), actions[:, i], label=f'action-agent{i}')

        # set opacity
        plt.setp(plt.gca().patches, alpha=0.3)


        # Adding vertical lines to indicate the steps
        for interval in intervals[i]:
            ax1.axvline(interval[1] + 1, color='grey', linestyle='--')

        # Adding shaded regions for the steps
        for interval in intervals[i]:
            ax1.axvspan(interval[0], interval[1] + 1, color='grey', alpha=0.1)

        plt.xlabel("Time")
        plt.ylabel("Action")
        plt.title(title)
        plt.legend(loc="upper left")
        # create x tick labels
        plt.xticks(np.arange(0, len(socs), 1))

        # plot price in the same figure but on the right y-axis
        plt.twinx()
        plt.plot(prices, label="Price", color='g')

        plt.ylabel("Price", color='g')
        plt.yticks(color='g')

        # # plot reward in the same figure but on the right y-axis
        # plt.twinx()
        # plt.plot(np.arange(0, len(rewards), 1), rewards[:, i], label=f'reward-agent{i}', color='y')
        #
        # plt.ylabel("Reward", color='y')
        # plt.yticks(color='y')

        # display legend on top right
        plt.legend(loc="upper right")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'plots/agent_{i}.png')
        plt.close(fig1)

    title = 'Total rewards'

    plt.plot(np.arange(0, len(socs), 1), rewards.sum(axis=1), label="total_rewards")

    # set opacity
    plt.setp(plt.gca().patches, alpha=0.3)

    fig1, ax1 = plt.subplots()

    for interval in intervals[i]:
        ax1.axvline(interval[1] + 1, color='grey', linestyle='--')

    # Adding shaded regions for the steps
    for interval in intervals[i]:
        ax1.axvspan(interval[0], interval[1] + 1, color='grey', alpha=0.1)


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

    # plt.show()
    plt.savefig(f'plots/total_rewards.png')
    plt.close(fig1)
