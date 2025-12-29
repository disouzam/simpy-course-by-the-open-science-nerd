import numpy as np
import simpy

import sensible_constants as sconst
from arrivals_generator import arrivals_generator


def single_run(
    experiment, rep=0, rc_period=sconst.RESULTS_COLLECTION_PERIOD, trace_enabled=False
):
    """
    Perform a single run of the model and return the results

    Parameters:
    -----------

    experiment: Experiment
        The experiment/paramaters to use with model
    """

    # results dictionary.  Each KPI is a new entry.
    run_results = {}

    # reset all result collection variables
    experiment.init_results_variables()

    # set random number set to the replication no.
    # this controls sampling for the run.
    experiment.set_random_no_set(rep)

    # environment is (re)created inside single run
    env = simpy.Environment()

    # we create simpy resource here - this has to be after we
    # create the environment object.
    experiment.operators = simpy.Resource(env, capacity=experiment.n_operators)

    # we pass the experiment to the arrivals generator
    env.process(arrivals_generator(env, experiment, trace_enabled))
    env.run(until=rc_period)

    # end of run results: calculate mean waiting time
    run_results["01_mean_waiting_time"] = np.mean(experiment.results["waiting_times"])

    # end of run results: calculate mean operator utilisation
    run_results["02_operator_util"] = (
        experiment.results["total_call_duration"] / (rc_period * experiment.n_operators)
    ) * 100.0

    run_results["03_total_call_duration"] = experiment.results["total_call_duration"]
    run_results["04_mean_call_duration"] = np.mean(experiment.results["call_durations"])

    # return the results from the run of the model
    return run_results
