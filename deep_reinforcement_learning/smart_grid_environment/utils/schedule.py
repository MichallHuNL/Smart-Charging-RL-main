import numpy as np


def calculate_schedule(schedule_shape, arrival_times, departure_times):
    schedule = np.zeros(schedule_shape)
    for arrival, departure in zip(arrival_times, departure_times):
        for port_idx in range(schedule_shape[0]):
            if schedule[port_idx, arrival] == 0:
                charging_time = departure - arrival
                for time_left in reversed(range(charging_time)):
                    if (arrival + charging_time - time_left) < schedule_shape[1]:
                        schedule[port_idx, arrival + charging_time - time_left] = time_left
                break
    return schedule
