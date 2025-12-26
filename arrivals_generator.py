"""
Implements an arrivals generator
"""

import itertools
from typing import Any, Generator

import numpy as np

from service_process import service


def arrivals_generator(env, operators) -> Generator[Any, Any, None]:
    """
    Simulates the call arrival process and spawns
    Inter-arrival time (IAT) is exponentially distributed

    Parameters:
    ------
    env: simpy.Environment
        The simpy environment for the simulation
    """
    arrivals_rng: np.random.Generator = np.random.default_rng()

    service_rng: np.random.Generator = np.random.default_rng()

    for caller_count in itertools.count(start=1):
        inter_arrival_time = arrivals_rng.exponential(60.0 / 100.0)
        yield env.timeout(inter_arrival_time)

        print(f"Call {caller_count} arrives at: {env.now:.2f}")

        env.process(
            service(
                identifier=caller_count,
                operators=operators,
                env=env,
                service_rng=service_rng,
            )
        )
