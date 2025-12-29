import numpy as np
import pandas as pd
import simpy
import simulation_constants as sim_const
from processes import arrivals_generator, warmup_complete


def single_run(
    experiment,
    rep=0,
    wu_period=sim_const.WARM_UP_PERIOD,
    rc_period=sim_const.RESULTS_COLLECTION_PERIOD,
):
    """
    Perform a single run of the model and return the results

    Parameters:
    -----------

    experiment: Experiment
        The experiment/paramaters to use with model

    rep: int
        The replication number.

    wu_period: float, optional (default=WARM_UP_PERIOD)
        The initial transient period of the simulation
        Results from this period are removed from final computations.

    rc_period: float, optional (default=RESULTS_COLLECTION_PERIOD)
        The run length of the model following warm up where results are
        collected.
    """

    # results dictionary.  Each KPI is a new entry.
    run_results = {}

    # reset all results variables to zero and empty
    experiment.init_results_variables()

    # set random number set to the replication no.
    # this controls sampling for the run.
    experiment.set_random_no_set(rep)

    # environment is (re)created inside single run
    env = simpy.Environment()

    # we create simpy resource here - this has to be after we
    # create the environment object.
    experiment.operators = simpy.Resource(env, capacity=experiment.n_operators)

    # #########################################################################
    # MODIFICATION: create the nurses resource
    experiment.nurses = simpy.Resource(env, capacity=experiment.n_nurses)
    # #########################################################################

    # we pass the experiment to the arrivals generator
    env.process(arrivals_generator(env, experiment))

    # #########################################################################
    # MODIFICATON: add warm-up period event
    env.process(warmup_complete(wu_period, env, experiment))

    # run for warm-up + results collection period
    env.run(until=wu_period + rc_period)
    # #########################################################################

    # end of run results: calculate mean waiting time
    run_results["01_mean_waiting_time"] = np.mean(experiment.results["waiting_times"])

    # end of run results: calculate mean operator utilisation
    run_results["02_operator_util"] = (
        experiment.results["total_call_duration"] / (rc_period * experiment.n_operators)
    ) * 100.0

    # #########################################################################
    # MODIFICATION: summary results for nurse process

    # end of run results: nurse waiting time
    run_results["03_mean_nurse_waiting_time"] = np.mean(
        experiment.results["nurse_waiting_times"]
    )

    # end of run results: calculate mean nurse utilisation
    run_results["04_nurse_util"] = (
        experiment.results["total_nurse_call_duration"]
        / (rc_period * experiment.n_nurses)
    ) * 100.0

    # #########################################################################

    # return the results from the run of the model
    return run_results


def multiple_replications(
    experiment,
    wu_period=sim_const.WARM_UP_PERIOD,
    rc_period=sim_const.RESULTS_COLLECTION_PERIOD,
    n_reps=5,
):
    """
    Perform multiple replications of the model.

    Params:
    ------
    experiment: Experiment
        The experiment/paramaters to use with model

    rc_period: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
        results collection period.
        the number of minutes to run the model to collect results

    n_reps: int, optional (default=5)
        Number of independent replications to run.

    Returns:
    --------
    pandas.DataFrame
    """

    # loop over single run to generate results dicts in a python list.
    results = [
        single_run(experiment, rep, wu_period, rc_period) for rep in range(n_reps)
    ]

    # format and return results in a dataframe
    df_results = pd.DataFrame(results)
    df_results.index = np.arange(1, len(df_results) + 1)
    df_results.index.name = "rep"
    return df_results
