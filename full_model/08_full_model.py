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
    # default resources
    N_OPERATORS = 13

    # ##############################################################################
    # MODIFICATION: number of nurses available
    N_NURSES = 10
    # ##############################################################################

    # default mean inter-arrival time (exp)
    MEAN_IAT = 60 / 100

    ## default service time parameters (triangular)
    CALL_LOW = 5.0
    CALL_MODE = 7.0
    CALL_HIGH = 10.0

    # ##############################################################################
    # MODIFICATION: nurse defaults

    # nurse uniform distribution parameters
    NURSE_CALL_LOW = 10.0
    NURSE_CALL_HIGH = 20.0

    # probability of a callback (parameter of Bernoulli)
    CHANCE_CALLBACK = 0.4

    # sampling settings - we now need 4 streams
    N_STREAMS = 4
    DEFAULT_RND_SET = 0
    # ##############################################################################

    # Boolean switch to simulation results as the model runs
    TRACE = False

    # run variables
    RESULTS_COLLECTION_PERIOD = 1000

    # ##############################################################################
    # MODIFICATON: added a warm-up period, by default we will not use it.
    WARM_UP_PERIOD = 0
    # ##############################################################################
    return (
        CALL_HIGH,
        CALL_LOW,
        CALL_MODE,
        CHANCE_CALLBACK,
        DEFAULT_RND_SET,
        MEAN_IAT,
        NURSE_CALL_HIGH,
        NURSE_CALL_LOW,
        N_NURSES,
        N_OPERATORS,
        N_STREAMS,
        RESULTS_COLLECTION_PERIOD,
        TRACE,
        WARM_UP_PERIOD,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Distribution classes

    We will define two additional distribution classes (`Uniform` and `Bernoulli`) to encapsulate the random number generation, parameters and random seeds used in the sampling.  Take a look at how they work.

    > You should be able to reuse these classes in your own simulation models.  It is actually not a lot of code, but it is useful to build up a code base that you can reuse with confidence in your own projects.
    """)
    return


@app.cell
def _(np):
    class Bernoulli:
        """
        Convenience class for the Bernoulli distribution.
        packages up distribution parameters, seed and random generator.

        The Bernoulli distribution is a special case of the binomial distribution
        where a single trial is conducted

        Use the Bernoulli distribution to sample success or failure.
        """

        def __init__(self, p, random_seed=None):
            """
            Constructor

            Params:
            ------
            p: float
                probability of drawing a 1

            random_seed: int | SeedSequence, optional (default=None)
                A random seed to reproduce samples.  If set to none then a unique
                sample is created.
            """
            self.rand = np.random.default_rng(seed=random_seed)
            self.p = p

        def sample(self, size=None):
            """
            Generate a sample from the exponential distribution

            Params:
            -------
            size: int, optional (default=None)
                the number of samples to return.  If size=None then a single
                sample is returned.

            Returns:
            -------
            float or np.ndarray (if size >=1)
            """
            return self.rand.binomial(n=1, p=self.p, size=size)

    return (Bernoulli,)


@app.cell
def _(np):
    class Uniform:
        """
        Convenience class for the Uniform distribution.
        packages up distribution parameters, seed and random generator.
        """

        def __init__(self, low, high, random_seed=None):
            """
            Constructor

            Params:
            ------
            low: float
                lower range of the uniform

            high: float
                upper range of the uniform

            random_seed: int | SeedSequence, optional (default=None)
                A random seed to reproduce samples.  If set to none then a unique
                sample is created.
            """
            self.rand = np.random.default_rng(seed=random_seed)
            self.low = low
            self.high = high

        def sample(self, size=None):
            """
            Generate a sample from the exponential distribution

            Params:
            -------
            size: int, optional (default=None)
                the number of samples to return.  If size=None then a single
                sample is returned.

            Returns:
            -------
            float or np.ndarray (if size >=1)
            """
            return self.rand.uniform(low=self.low, high=self.high, size=size)

    return (Uniform,)


@app.cell
def _(np):
    class Triangular:
        """
        Convenience class for the triangular distribution.
        packages up distribution parameters, seed and random generator.
        """

        def __init__(self, low, mode, high, random_seed=None):
            """
            Constructor. Accepts and stores parameters of the triangular dist
            and a random seed.

            Params:
            ------
            low: float
                The smallest values that can be sampled

            mode: float
                The most frequently sample value

            high: float
                The highest value that can be sampled

            random_seed: int | SeedSequence, optional (default=None)
                Used with params to create a series of repeatable samples.
            """
            self.rand = np.random.default_rng(seed=random_seed)
            self.low = low
            self.high = high
            self.mode = mode

        def sample(self, size=None):
            """
            Generate one or more samples from the triangular distribution

            Params:
            --------
            size: int
                the number of samples to return.  If size=None then a single
                sample is returned.

            Returns:
            -------
            float or np.ndarray (if size >=1)
            """
            return self.rand.triangular(self.low, self.mode, self.high, size=size)

    return (Triangular,)


@app.cell
def _(np):
    class Exponential:
        """
        Convenience class for the exponential distribution.
        packages up distribution parameters, seed and random generator.
        """

        def __init__(self, mean, random_seed=None):
            """
            Constructor

            Params:
            ------
            mean: float
                The mean of the exponential distribution

            random_seed: int| SeedSequence, optional (default=None)
                A random seed to reproduce samples.  If set to none then a unique
                sample is created.
            """
            self.rand = np.random.default_rng(seed=random_seed)
            self.mean = mean

        def sample(self, size=None):
            """
            Generate a sample from the exponential distribution

            Params:
            -------
            size: int, optional (default=None)
                the number of samples to return.  If size=None then a single
                sample is returned.

            Returns:
            -------
            float or np.ndarray (if size >=1)
            """
            return self.rand.exponential(self.mean, size=size)

    return (Exponential,)


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
def _(
    Bernoulli,
    CALL_HIGH,
    CALL_LOW,
    CALL_MODE,
    CHANCE_CALLBACK,
    DEFAULT_RND_SET,
    Exponential,
    MEAN_IAT,
    NURSE_CALL_HIGH,
    NURSE_CALL_LOW,
    N_NURSES,
    N_OPERATORS,
    N_STREAMS,
    Triangular,
    Uniform,
    np,
):
    class Experiment:
        """
        Encapsulates the concept of an experiment 🧪 with the urgent care
        call centre simulation model.

        An Experiment:
        1. Contains a list of parameters that can be left as defaults or varied
        2. Provides a place for the experimentor to record results of a run
        3. Controls the set & streams of pseudo random numbers used in a run.

        """

        def __init__(
            self,
            random_number_set=DEFAULT_RND_SET,
            n_streams=N_STREAMS,
            n_operators=N_OPERATORS,
            mean_iat=MEAN_IAT,
            call_low=CALL_LOW,
            call_mode=CALL_MODE,
            call_high=CALL_HIGH,
            # ######################################################################
            # MODIFICATION: nurse parameters
            n_nurses=N_NURSES,
            chance_callback=CHANCE_CALLBACK,
            nurse_call_low=NURSE_CALL_LOW,
            nurse_call_high=NURSE_CALL_HIGH,
            ########################################################################
        ):
            """
            The init method sets up our defaults.
            """
            # sampling
            self.random_number_set = random_number_set
            self.n_streams = n_streams

            # store parameters for the run of the model
            self.n_operators = n_operators
            self.mean_iat = mean_iat
            self.call_low = call_low
            self.call_mode = call_mode
            self.call_high = call_high

            # resources: we must init resources after an Environment is created.
            # But we will store a placeholder for transparency
            self.operators = None

            # ######################################################################
            # MODIFICATION: nurse parameters
            self.n_nurses = n_nurses
            self.chance_callback = chance_callback
            self.nurse_call_low = nurse_call_low
            self.nurse_call_high = nurse_call_high

            # nurse resources placeholder
            self.nurses = None
            # ######################################################################

            # initialise results to zero
            self.init_results_variables()

            # initialise sampling objects
            self.init_sampling()

        def set_random_no_set(self, random_number_set):
            """
            Controls the random sampling
            Parameters:
            ----------
            random_number_set: int
                Used to control the set of pseudo random numbers used by
                the distributions in the simulation.
            """
            self.random_number_set = random_number_set
            self.init_sampling()

        def init_sampling(self):
            """
            Create the distributions used by the model and initialise
            the random seeds of each.
            """
            # produce n non-overlapping streams
            seed_sequence = np.random.SeedSequence(self.random_number_set)
            self.seeds = seed_sequence.spawn(self.n_streams)

            # create distributions

            # call inter-arrival times
            self.arrival_dist = Exponential(self.mean_iat, random_seed=self.seeds[0])

            # duration of call triage
            self.call_dist = Triangular(
                self.call_low,
                self.call_mode,
                self.call_high,
                random_seed=self.seeds[1],
            )

            # ######################################################################
            # MODIFICATION create the callback and nurse consultation distributions
            self.callback_dist = Bernoulli(
                self.chance_callback, random_seed=self.seeds[2]
            )

            self.nurse_dist = Uniform(
                self.nurse_call_low,
                self.nurse_call_high,
                random_seed=self.seeds[3],
            )
            # ######################################################################

        def init_results_variables(self):
            """
            Initialise all of the experiment variables used in results
            collection.  This method is called at the start of each run
            of the model
            """
            # variable used to store results of experiment
            self.results = {}
            self.results["waiting_times"] = []

            # total operator usage time for utilisation calculation.
            self.results["total_call_duration"] = 0.0

            # ######################################################################
            # MODIFICATION: nurse sub process results collection
            self.results["nurse_waiting_times"] = []
            self.results["total_nurse_call_duration"] = 0.0
            # ######################################################################

    return (Experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Modified model code

    We will modify the model code and logic that we have already developed to include a nurse consultation for a proportion of the callers.  We create a new function called `nurse_consultation` that contains all the logic. We also need to modify the `service` function so that a proportion of calls are sent to the nurse consultation process.
    """)
    return


@app.cell
def _(TRACE):
    def trace(msg):
        """
        Turing printing of events on and off.

        Params:
        -------
        msg: str
            string to print to screen.
        """
        if TRACE:
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
def _(
    RESULTS_COLLECTION_PERIOD,
    WARM_UP_PERIOD,
    arrivals_generator,
    np,
    simpy,
    warmup_complete,
):
    def single_run(
        experiment, rep=0, wu_period=WARM_UP_PERIOD, rc_period=RESULTS_COLLECTION_PERIOD
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
def _(RESULTS_COLLECTION_PERIOD, WARM_UP_PERIOD, np, pd, single_run):
    def multiple_replications(
        experiment,
        wu_period=WARM_UP_PERIOD,
        rc_period=RESULTS_COLLECTION_PERIOD,
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
