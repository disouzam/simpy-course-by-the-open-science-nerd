def service(identifier, operators, env, service_rng):
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
        yield req

        waiting_time = env.now - start_wait
        print(f"Operator answered call {identifier} at {env.now:.2f}")

        call_duration = service_rng.triangular(left=5.0, mode=7.0, right=10.0)
        yield env.timeout(call_duration)

        print(
            f"call {identifier} ended {env.now:.2f}; "
            + f"waiting time was {waiting_time:.2f}"
        )
