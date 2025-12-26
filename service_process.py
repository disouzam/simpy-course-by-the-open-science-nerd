"""
Implements a service process
"""

from typing import Any, Generator

from colored import Back, Fore, Style

EPSILON = 1e-3


def service(identifier, operators, env, service_rng) -> Generator[Any, Any, None]:
    """
    Simulates the service process for a call operator

    1. Request and wait for a call operator
    2. Phone triage (triangular distribution)
    3. Exit system

    Params:
    ------

    identifier: int
        A unique identifier for this caller

    operators: simpy.Resource
        The pool of call operators that answer calls
        These are shared across resources.

    env: simpy.Environment
        The current environment the simulation is running in
        We use this to pause and restart the process after a delay

    service_rng:: numpy.random.Generator:
        The random number generator used to sample service times
    """
    start_wait = env.now

    with operators.request() as req:
        active_operators = operators.count
        remaining_operators = operators.capacity - active_operators
        yield req

        waiting_time = env.now - start_wait

        main_message = f"Call {Fore.white}{Back.black} {identifier} {Style.reset} answered by operator at {env.now:.2f} - Active operators: {active_operators} - Remaining operators: {remaining_operators}"
        if waiting_time < EPSILON:
            print(f"{main_message} - Immediate response\n")
        else:
            print(
                f"{main_message} -  {Fore.white}{Back.red} Waiting time was {waiting_time:.2f}{Style.reset}\n"
            )

        call_duration = service_rng.triangular(left=5.0, mode=7.0, right=10.0)
        yield env.timeout(call_duration)

        print(
            f"\n{Fore.black}{Back.green}Call {Fore.white}{Back.black} {identifier} {Fore.black}{Back.green} ended at {env.now:.2f} - Duration was: {call_duration:.2f}{Style.reset}\n"
        )
