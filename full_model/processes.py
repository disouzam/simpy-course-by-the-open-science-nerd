import itertools

from tracing import trace


def nurse_consultation(identifier, env, args):
    """
    simulates the wait for an consultation with a nurse on the phone.

    1. request and wait for a nurse resource
    2. phone consultation (uniform)
    3. release nurse and exit system

    """
    trace(f"Patient {identifier} waiting for nurse call back")
    start_nurse_wait = env.now

    # request a nurse
    with args.nurses.request() as req:
        yield req

        # record the waiting time for nurse call back
        nurse_waiting_time = env.now - start_nurse_wait
        args.results["nurse_waiting_times"].append(nurse_waiting_time)

        # sample nurse the duration of the nurse consultation
        nurse_call_duration = args.nurse_dist.sample()

        trace(f"nurse called back patient {identifier} at " + f"{env.now:.3f}")

        # schedule process to begin again after call duration
        yield env.timeout(nurse_call_duration)

        args.results["total_nurse_call_duration"] += nurse_call_duration

        trace(f"nurse consultation for {identifier}" + f" competed at {env.now:.3f}")


def service(identifier, env, args):
    """
    simulates the service process for a call operator

    1. request and wait for a call operator
    2. phone triage (triangular)
    3. release call operator
    4. a proportion of call continue to nurse consultation

    Params:
    ------
    identifier: int
        A unique identifier for this caller

    env: simpy.Environment
        The current environment the simulation is running in
        We use this to pause and restart the process after a delay.

    args: Experiment
        The settings and input parameters for the current experiment

    """

    # record the time that call entered the queue
    start_wait = env.now

    # request an operator - stored in the Experiment
    with args.operators.request() as req:
        yield req

        # record the waiting time for call to be answered
        waiting_time = env.now - start_wait

        # store the results for an experiment
        args.results["waiting_times"].append(waiting_time)
        trace(f"operator answered call {identifier} at " + f"{env.now:.3f}")

        # the sample distribution is defined by the experiment.
        call_duration = args.call_dist.sample()

        # schedule process to begin again after call_duration
        yield env.timeout(call_duration)

        # update the total call_duration
        args.results["total_call_duration"] += call_duration

        # print out information for patient.
        trace(
            f"call {identifier} ended {env.now:.3f}; "
            + f"waiting time was {waiting_time:.3f}"
        )

    # ##########################################################################
    # MODIFICATION NURSE CALL BACK
    # does nurse need to call back?
    # Note the level of the indented code.
    callback_patient = args.callback_dist.sample()

    if callback_patient:
        env.process(nurse_consultation(identifier, env, args))
    # ##########################################################################


def arrivals_generator(env, args):
    """
    IAT is exponentially distributed

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
        # rhe sample distribution is defined by the experiment.
        inter_arrival_time = args.arrival_dist.sample()
        yield env.timeout(inter_arrival_time)

        trace(f"call arrives at: {env.now:.3f}")

        # create a service process
        env.process(service(caller_count, env, args))
