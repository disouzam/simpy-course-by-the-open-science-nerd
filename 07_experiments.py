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
    # Experiments 🧪

    A key part of any computer simulation study is **experimentation**.  Here a set of experiments will be conducted in an attempt to understand and find improvements to the system under study.  Experiments essentially vary inputs and process logic.

    We can do this manually, but as we develop a model the number of input parameters will increase.

    💡 There are several data structures you might employ to organise parameters.

    * a python dictionary
    * a custom parameter class
    * a dataclass

    All of these approaches work well and it really is a matter of judgement on what you prefer. One downside of a python `dict` and a custom class is that they are both mutable (although a class can have custom properties where users can only access 'viewable' attributes).  A dataclass can easily be made immutable and requires less code than a custom class, but has the downside that its syntax is a little less pythonic. Here we will build a parameter class called `Experiment`.

    > ☺️ We will also use this re-organisation of code to eliminate our global variables!
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
    import simpy

    return itertools, np, simpy


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

    # default mean inter-arrival time (exp)
    MEAN_IAT = 60 / 100

    # default service time parameters (triangular)
    CALL_LOW = 5.0
    CALL_MODE = 7.0
    CALL_HIGH = 10.0

    # sampling settings
    N_STREAMS = 2
    DEFAULT_RND_SET = 0

    # Boolean switch to simulation results as the model runs
    TRACE = False

    # run variables
    RESULTS_COLLECTION_PERIOD = 1000
    return (
        CALL_HIGH,
        CALL_LOW,
        CALL_MODE,
        DEFAULT_RND_SET,
        MEAN_IAT,
        N_OPERATORS,
        N_STREAMS,
        RESULTS_COLLECTION_PERIOD,
        TRACE,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Distribution classes

    We will define two distribution classes (`Triangular` and `Exponential`) to encapsulate the random number generation, parameters and random seeds used in the sampling.  This simplifies what we will need to include in the `Experiment` class and as we will see later makes it easier to vary distributions as well as parameters.
    """)
    return


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

    An experiment class is useful because it allows use to easily configure and schedule a large number of experiments to occur in a loop.  We set the class up so that it uses the default variables we defined above i.e. as default the model reflects the as-is process.  To run a new experiment we simply override the default values.
    """)
    return


@app.cell
def _(
    CALL_HIGH,
    CALL_LOW,
    CALL_MODE,
    DEFAULT_RND_SET,
    Exponential,
    MEAN_IAT,
    N_OPERATORS,
    N_STREAMS,
    Triangular,
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
            n_operators=N_OPERATORS,
            mean_iat=MEAN_IAT,
            call_low=CALL_LOW,
            call_mode=CALL_MODE,
            call_high=CALL_HIGH,
            n_streams=N_STREAMS,
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

    return (Experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1. Creating a default experiment
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To use `Experiment` is very simple.  For example to create a default experiment (that uses all the default parameter values) we would use the following code
    """)
    return


@app.cell
def _(Experiment, simpy):
    env = simpy.Environment()
    _ = env
    default_experiment = Experiment()
    return (default_experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Due to the way python works we can access all of the experiment variables from the `default_scenario` object. For example the following code will generate an inter-arrival time:
    """)
    return


@app.cell
def _(default_experiment):
    default_experiment.arrival_dist.sample()
    return


@app.cell
def _(default_experiment):
    default_experiment.mean_iat
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Creating an experiment with more call operators

    To change parameters in an experiment we just need to include a new value when we create the `Experiment`.  For example if we wanted to increase the number of servers to 14. We use the following code:
    """)
    return


@app.cell
def _(Experiment, simpy):
    env_2 = simpy.Environment()
    _ = env_2
    extra_server = Experiment(n_operators=14)
    return (extra_server,)


@app.cell
def _(extra_server):
    extra_server.n_operators
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Modified model code

    We will modify the model code and logic that we have already developed.  The functions for service and arrivals will now accept an `Experiment` argument.

    > Note that at this point you could put all of the code into a python module and import the functions and classes you need into an experiment workbook.
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
    def service(identifier, env, args):
        """
        simulates the service process for a call operator

        1. request and wait for a call operator
        2. phone triage (triangular)
        3. exit system

        Params:
        ------

        identifier: int
            A unique identifier for this caller

        env: simpy.Environment
            The current environent the simulation is running in
            We use this to pause and restart the process after a delay.

        args: Experiment
            The settings and input parameters for the current experiment

        """

        # record the time that call entered the queue
        start_wait = env.now

        # MODIFICATION: request an operator - stored in the Experiment
        with args.operators.request() as req:
            yield req

            # record the waiting time for call to be answered
            waiting_time = env.now - start_wait

            # ######################################################################
            # MODIFICATION: store the results for an experiment
            args.results["waiting_times"].append(waiting_time)
            # ######################################################################

            trace(f"operator answered call {identifier} at " + f"{env.now:.3f}")

            # ######################################################################
            # MODIFICATION: the sample distribution is defined by the experiment.
            call_duration = args.call_dist.sample()
            # ######################################################################

            # schedule process to begin again after call_duration
            yield env.timeout(call_duration)

            # update the total call_duration
            args.results["total_call_duration"] += call_duration

            # print out information for patient.
            trace(
                f"call {identifier} ended {env.now:.3f}; "
                + f"waiting time was {waiting_time:.3f}"
            )

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
            # ######################################################################
            # MODIFICATION:the sample distribution is defined by the experiment.
            inter_arrival_time = args.arrival_dist.sample()
            ########################################################################

            yield env.timeout(inter_arrival_time)

            trace(f"call arrives at: {env.now:.3f}")

            # ######################################################################
            # MODIFICATION: we pass the experiment to the service function
            env.process(service(caller_count, env, args))
            # ######################################################################

    return (arrivals_generator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. A single run wrapper function
    """)
    return


@app.cell
def _(RESULTS_COLLECTION_PERIOD, arrivals_generator, np, simpy):
    def single_run(experiment, rep=0, rc_period=RESULTS_COLLECTION_PERIOD):
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
        env.process(arrivals_generator(env, experiment))
        env.run(until=rc_period)

        # end of run results: calculate mean waiting time
        run_results["01_mean_waiting_time"] = np.mean(
            experiment.results["waiting_times"]
        )

        # end of run results: calculate mean operator utilisation
        run_results["02_operator_util"] = (
            experiment.results["total_call_duration"]
            / (rc_period * experiment.n_operators)
        ) * 100.0

        # return the results from the run of the model
        return run_results

    return (single_run,)


@app.cell
def _(Experiment, single_run):
    TRACE = False
    default_scenario = Experiment()
    results = single_run(default_scenario)
    print(
        f"Mean waiting time: {results['01_mean_waiting_time']:.2f} mins \n"
        + f"Operator Utilisation {results['02_operator_util']:.2f}%"
    )
    return (TRACE,)


if __name__ == "__main__":
    app.run()
