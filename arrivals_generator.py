import itertools

import numpy as np

from service_process import service


def arrivals_generator(env, operators):
    """
    Simulates the call arrival process and spawns
    Inter-arrival time (IAT) is exponentially distributed

    Parameters:
    ------
    env: simpy.Environment
        The simpy environment for the simulation
    """
    arrivals_rng = np.random.default_rng()

    service_rng = np.random.default_rng()

    for caller_count in itertools.count(start=1):
        inter_arrival_time = arrivals_rng.exponential(60.0 / 100.0)
        yield env.timeout(inter_arrival_time)

        print(f"Call {caller_count} arrives at: {env.now:.2f}")

        env.process(service(caller_count, operators, env, service_rng))
