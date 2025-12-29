"""
Implements an arrivals generator
"""

import itertools
from typing import Any, Generator

from colored import Back, Fore, Style

from logging_and_tracing import trace
from service_process import service


def arrivals_generator(env, args, trace_enabled=False) -> Generator[Any, Any, None]:
    """
    Simulates the call arrival process and spawns
    Inter-arrival time (IAT) is exponentially distributed

    Parameters:
    ------
    env: simpy.Environment
        The simpy environment for the simulation

    args: Experiment
        The settings and input parameters for the simulation.
    """
    # use itertools as it provides an infinite loop
    # with a counter variable that we can use for unique Ids
    for caller_count in itertools.count(start=1):
        # ######################################################################
        # MODIFICATION:the sample distribution is defined by the experiment.
        inter_arrival_time = args.arrival_dist.sample()
        ########################################################################

        yield env.timeout(inter_arrival_time)

        trace(
            f"{Fore.blue}Call {Fore.white}{Back.black} {caller_count} {Style.reset} {Fore.blue} arrives at: {env.now:.2f}{Style.reset}",
            enabled=trace_enabled,
        )

        # ######################################################################
        # MODIFICATION: we pass the experiment to the service function
        env.process(
            service(
                identifier=caller_count, env=env, args=args, trace_enabled=trace_enabled
            )
        )
        # ######################################################################
