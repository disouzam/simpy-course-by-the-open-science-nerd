import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Notice of authorship

    Most content in this notebook has been copied / typed based on corresponding code written / made available by Tom Monks (The Open Science Nerd)

    **Repository**: https://github.com/pythonhealthdatascience/intro-open-sim

    **YouTube channel**: https://www.youtube.com/@TheOpenScienceNerd

    **Simpy short course**: https://youtube.com/playlist?list=PLrOeiVQ0eMwF6qE5RLs2brgxBfVUy3MO3&si=ZTorhOyZsbi8XftM
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The full urgent care centre call model

    **We can now update the model to include the final call centre logic!**

    After a patient has spoken to a call operator their priority is triaged.  It is estimated that 40% of patients require a callback from a nurse.  There are 10 nurses available.  A nurse patient consultation has a Uniform distribution lasting between 10 and 20 minutes.

    > ⏰ Some call centres run 24/7 while others are open for specified time window during the day.  So we need to cope with **both terminating and non-terminating systems**.

    ![model image](public/full_model.png "Urgent care call centre")

    **Modifications needed**

    * Add new default variables for the nurse consultation parameters
    * Add new decision variables to `Experiment` for the no. nurses and the consultation distribution.
    * Create a second `simpy.Resource` called `nurses` and add it to the simulation model.
    * Create the nurse consultation process
    * Modify the logic of `service` so that % of patients are called back.
    * Collect results and estimate the waiting time for a nurse consultation and nurse utilisation.
    * **Bonus:** Add an *optional* warm-up period event.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports
    """)
    return


@app.cell
def _():
    import itertools

    import numpy as np
    import pandas as pd
    import simpy

    return itertools, np, pd, simpy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Notebook level variables, constants, and default values

    A useful first step when setting up a simulation model is to define the base case or as-is parameters.  Here we will create a set of constant/default values for our `Experiment` class, but you could also consider reading these in from a file.
    """)
    return


@app.cell
def _():
    import simulation_constants as sim_const

    return (sim_const,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Distribution classes

    We will define two additional distribution classes (`Uniform` and `Bernoulli`) to encapsulate the random number generation, parameters and random seeds used in the sampling.  Take a look at how they work.

    > You should be able to reuse these classes in your own simulation models.  It is actually not a lot of code, but it is useful to build up a code base that you can reuse with confidence in your own projects.
    """)
    return


@app.cell
def _():
    # from probability_distributions import Bernoulli, Exponential, Triangular, Uniform
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Experiment class

    We will modify the experiment class to include new results collection for the additional nurse process.

    1. Modify the __init__ method to accept additional parameters: `chance_callback`, `nurse_call_low`, `nurse_call_high`. Remember to include the default values for these parameters.
    2. Store parameters in the class and create new distributions.
    3. Add variables to support KPI calculation to the `results` dictionary for `nurse_waiting_times` and `total_nurse_call_duration`
    """)
    return


@app.cell
def _():
    from experiment import Experiment

    return (Experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Modified model code

    We will modify the model code and logic that we have already developed to include a nurse consultation for a proportion of the callers.  We create a new function called `nurse_consultation` that contains all the logic. We also need to modify the `service` function so that a proportion of calls are sent to the nurse consultation process.
    """)
    return


@app.cell
def _(sim_const):
    def trace(msg):
        """
        Turing printing of events on and off.

        Params:
        -------
        msg: str
            string to print to screen.
        """
        if sim_const.TRACE:
            print(msg)

    return (trace,)


@app.cell
def _(trace):
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

            trace(
                f"nurse consultation for {identifier}" + f" competed at {env.now:.3f}"
            )

    return (nurse_consultation,)


@app.cell
def _(nurse_consultation, trace):
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

    return (service,)


@app.cell
def _(itertools, service, trace):
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

    return (arrivals_generator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🥵 Warm-up period

    The call centre model starts from empty.  If the call centre runs 24/7 then it is a non-terminating system and our estimates of waiting time and server utilisation are biased due to the empty period at the start of the simulation.  We can remove this initialisation bias using a warm-up period.

    We will implement a warm-up through an **event** that happens once in a single run of the model.  The model will be run for the **warm-up period + results collection period**.  At the end of the warm-up period an event will happen where all variables in the current experiment are reset (e.g. empty lists and set quantitative values to 0.0).

    > **Note**: at the point results are reset there are likely resources (call operators and nurses) in use. The result is that we carry over some of the resource usage time from the warm-up to results collection period. It isn't a big deal, but there's potential for resource usage time to be slightly higher than the time scheduled.
    """)
    return


@app.cell
def _(trace):
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

    return (warmup_complete,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Model wrapper functions

    Modifications to make to the `single_run` function:

    1. Add a warm-up parameters called `wu_period`
    1. Create and the nurses resource to the experiment
    2. Schedule the `warm_up_complete` process.
    3. After the simulation is complete calculate the mean waiting time and mean utilisation for nurses.
    """)
    return


@app.cell
def _(arrivals_generator, np, sim_const, simpy, warmup_complete):
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
        run_results["01_mean_waiting_time"] = np.mean(
            experiment.results["waiting_times"]
        )

        # end of run results: calculate mean operator utilisation
        run_results["02_operator_util"] = (
            experiment.results["total_call_duration"]
            / (rc_period * experiment.n_operators)
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

    return (single_run,)


@app.cell
def _(np, pd, sim_const, single_run):
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

    return (multiple_replications,)


@app.cell
def _(Experiment, multiple_replications):
    scenario = Experiment(n_nurses=15, nurse_call_high=30.0)
    results = multiple_replications(scenario, wu_period=50.0)
    results.describe().round(1).T
    return


if __name__ == "__main__":
    app.run()
