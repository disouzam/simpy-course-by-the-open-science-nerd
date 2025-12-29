import itertools

import simulation_constants as sim_const
from colored import Back, Fore, Style
from tracing import trace


def nurse_consultation(identifier, env, args):
    """
    simulates the wait for an consultation with a nurse on the phone.

    1. Request and wait for a nurse resource
    2. Phone consultation (uniform distribution)
    3. Release nurse and exit system

    """
    trace_enabled = True
    trace(
        f"{Fore.black}{Back.yellow}Patient {Fore.white}{Back.black} {identifier} {Fore.black}{Back.yellow} waiting for nurse call back{Style.reset}",
        enabled=trace_enabled,
    )
    start_nurse_wait = env.now

    # request a nurse
    with args.nurses.request() as req:
        active_nurses = args.nurses.count
        remaining_nurses = args.nurses.capacity - active_nurses
        yield req

        # record the waiting time for nurse call back
        nurse_waiting_time = env.now - start_nurse_wait
        args.results["nurse_waiting_times"].append(nurse_waiting_time)

        # sample nurse the duration of the nurse consultation
        nurse_call_duration = args.nurse_dist.sample()
        main_message = (
            f"\n{Fore.dark_orange}{Back.grey_0}Nurse called back patient {Fore.white}{Back.black} {identifier} {Fore.dark_orange}{Back.grey_0} at "
            + f"{env.now:.2f} - Active nurses: {active_nurses} - Remaining nurses: {remaining_nurses}{Style.reset}"
        )

        if nurse_waiting_time < sim_const.EPSILON:
            trace(
                f"{main_message} - Immediate response",
                enabled=trace_enabled,
            )
        else:
            trace(
                f"{main_message} -  {Fore.white}{Back.red} Waiting time was {nurse_waiting_time:.2f}{Style.reset}",
                trace_enabled,
            )

        # schedule process to begin again after call duration
        yield env.timeout(nurse_call_duration)

        args.results["total_nurse_call_duration"] += nurse_call_duration

        trace(
            f"\n{Fore.dark_orange}{Back.grey_0}Nurse consultation for {Fore.white}{Back.black} {identifier} {Fore.dark_orange}{Back.grey_0}"
            + f" completed at {env.now:.2f} {Style.reset}",
            enabled=trace_enabled,
        )


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
        active_operators = args.operators.count
        remaining_operators = args.operators.capacity - active_operators
        yield req

        # record the waiting time for call to be answered
        waiting_time = env.now - start_wait

        # store the results for an experiment
        args.results["waiting_times"].append(waiting_time)

        main_message = f"\nCall {Fore.white}{Back.black} {identifier} {Style.reset} answered by operator at {env.now:.2f} - Active operators: {active_operators} - Remaining operators: {remaining_operators}"

        trace_enabled = True
        if waiting_time < sim_const.EPSILON:
            trace(f"{main_message} - Immediate response", trace_enabled)
        else:
            trace(
                f"{main_message} -  {Fore.white}{Back.red} Waiting time was {waiting_time:.2f}{Style.reset}",
                trace_enabled,
            )

        # the sample distribution is defined by the experiment.
        call_duration = args.call_dist.sample()

        # schedule process to begin again after call_duration
        yield env.timeout(call_duration)

        # update the total call_duration
        args.results["total_call_duration"] += call_duration

        # print out information for patient.
        trace(
            f"\n{Fore.black}{Back.green}Call {Fore.white}{Back.black} {identifier} {Fore.black}{Back.green} ended at {env.now:.2f} - Duration was: {call_duration:.2f}{Style.reset}\n",
            trace_enabled,
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

        trace_enabled = True
        trace(
            f"\n{Fore.blue}Call {Fore.white}{Back.black} {caller_count} {Style.reset} {Fore.blue} arrives at: {env.now:.2f}{Style.reset}",
            enabled=trace_enabled,
        )

        # create a service process
        env.process(service(caller_count, env, args))


def warmup_complete(warm_up_period, env, args):
    """
    End of warm-up period event. Used to reset results collection variables.

    Parameters:
    ----------
    warm_up_period: float
        Duration of warm-up period in simultion time units

    env: simpy.Environment
        The simpy environment

    args: Experiment
        The simulation experiment that contains the results being collected.
    """
    yield env.timeout(warm_up_period)
    trace(f"{env.now:.2f}: Warm up complete.")

    args.init_results_variables()
