"""
Implements a service process
"""

from typing import Any, Generator

from colored import Back, Fore, Style

import sensible_constants as sconst
from logging_and_tracing import trace


def service(identifier, env, args, trace_enabled=False) -> Generator[Any, Any, None]:
    """
    Simulates the service process for a call operator

    1. Request and wait for a call operator
    2. Phone triage (triangular distribution)
    3. Exit system

    Params:
    ------

    identifier: int
        A unique identifier for this caller


    env: simpy.Environment
        The current environment the simulation is running in
        We use this to pause and restart the process after a delay

    service_rng: numpy.random.Generator
        The random number generator used to sample service times

    args: Experiment
        The settings and input parameters for the current experiment

    """
    # record the time that call entered the queue
    start_wait = env.now

    # MODIFICATION: request an operator - stored in the Experiment
    with args.operators.request() as req:
        active_operators = args.operators.count
        remaining_operators = args.operators.capacity - active_operators
        yield req

        # record the waiting time for call to be answered
        waiting_time = env.now - start_wait

        if args.results is not None:
            # ######################################################################
            # MODIFICATION: store the results for an experiment
            args.results["waiting_times"].append(waiting_time)
            # ######################################################################

        main_message = f"Call {Fore.white}{Back.black} {identifier} {Style.reset} answered by operator at {env.now:.2f} - Active operators: {active_operators} - Remaining operators: {remaining_operators}"

        if waiting_time < sconst.EPSILON:
            trace(f"{main_message} - Immediate response\n", trace_enabled)
        else:
            trace(
                f"{main_message} -  {Fore.white}{Back.red} Waiting time was {waiting_time:.2f}{Style.reset}\n",
                trace_enabled,
            )

        # ######################################################################
        # MODIFICATION: the sample distribution is defined by the experiment.
        # sample call duration
        call_duration = args.call_dist.sample()
        # ######################################################################

        # schedule process to begin again after call_duration
        yield env.timeout(call_duration)

        # update the total call_duration
        args.results["call_durations"].append(call_duration)
        args.results["total_call_duration"] += call_duration

        # print out information for patient.
        trace(
            f"\n{Fore.black}{Back.green}Call {Fore.white}{Back.black} {identifier} {Fore.black}{Back.green} ended at {env.now:.2f} - Duration was: {call_duration:.2f}{Style.reset}\n",
            trace_enabled,
        )
