import numpy as np


def calculate_schedule(schedule_shape, arrival_times, departure_times):
    schedule = np.zeros(schedule_shape)
    ends = np.zeros(schedule_shape)

    # Sort based on arrival times
    zipped = sorted(zip(arrival_times, departure_times), key=lambda x: x[0])

    for arrival, departure in zipped:
        for port_idx in range(schedule_shape[0]):
            if schedule[port_idx, arrival] == 0 and (arrival == 0 or schedule[port_idx, arrival - 1] == 0):
                charging_time = departure - arrival
                for time_left in reversed(range(charging_time + 1)):
                    if (arrival + charging_time - time_left) < schedule_shape[1]:
                        schedule[port_idx, arrival + charging_time - time_left] = time_left
                        if time_left == 0:
                            ends[port_idx, arrival + charging_time] = 1
                break
    return schedule, ends
