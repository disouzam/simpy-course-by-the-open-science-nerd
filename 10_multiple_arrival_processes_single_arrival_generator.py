import marimo

__generated_with = "0.19.4"
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
    ## A single arrival generator function

    We can write less code by modifying the `Experiment` class to use `dict` to store the distributions and by creating a generator function that accepts parameters.

    The code is more flexible, at the cost of readability (and potential to make more mistakes) for some less experienced coders.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Imports
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
    # 2. Constants
    """)
    return


@app.cell
def _():
    # default mean inter-arrival times (exponential distribution)
    IAT_SHOULDER = 24.0
    IAT_HIP = 32.0
    IAT_WRIST = 21.0
    IAT_ANKLE = 17.0

    # sampling settings
    N_STREAMS = 4
    DEFAULT_RND_SET = 0

    # run variables (units = hours)
    RUN_LENGTH = 24 * 10
    return (
        DEFAULT_RND_SET,
        IAT_ANKLE,
        IAT_HIP,
        IAT_SHOULDER,
        IAT_WRIST,
        N_STREAMS,
        RUN_LENGTH,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Helper classes and functions
    """)
    return


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


@app.function
def trace(msg, trace_enabled: False):
    """
    Turing printing of events on and off.

    Params:
    -------
    msg: str
        string to print to screen.
    """
    if trace_enabled:
        print(msg)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Experiment class
    """)
    return


@app.cell
def _(
    DEFAULT_RND_SET,
    Exponential,
    IAT_ANKLE,
    IAT_HIP,
    IAT_SHOULDER,
    IAT_WRIST,
    N_STREAMS,
    np,
):
    class Experiment:
        """
        Encapsulates the concept of an experiment 🧪 for the Orthopedic Surgey
        trauma arrival simulator. Manages parameters, PRNG streams and results.
        """

        def __init__(
            self,
            random_number_set=DEFAULT_RND_SET,
            n_streams=N_STREAMS,
            iat_shoulder=IAT_SHOULDER,
            iat_hip=IAT_HIP,
            iat_wrist=IAT_WRIST,
            iat_ankle=IAT_ANKLE,
        ):
            """
            The init method sets up our defaults.
            """
            # sampling
            self.random_number_set = random_number_set
            self.n_streams = n_streams

            # store parameters for the run of the model
            self.iat_shoulder = iat_shoulder
            self.iat_hip = iat_hip
            self.iat_wrist = iat_wrist
            self.iat_ankle = iat_ankle

            # we will store all code in distributions
            self.dists = {}

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

            # inter-arrival time distributions
            self.dists["shoulder"] = Exponential(
                self.iat_shoulder, random_seed=self.seeds[0]
            )

            self.dists["hip"] = Exponential(self.iat_hip, random_seed=self.seeds[1])

            self.dists["wrist"] = Exponential(self.iat_wrist, random_seed=self.seeds[2])

            self.dists["ankle"] = Exponential(self.iat_ankle, random_seed=self.seeds[3])

        def init_results_variables(self):
            """
            Initialise all of the experiment variables used in results
            collection.  This method is called at the start of each run
            of the model
            """
            # variable used to store results of experiment
            self.results = {}
            self.results["n_shoulders"] = 0
            self.results["n_hips"] = 0
            self.results["n_wrists"] = 0
            self.results["n_ankles"] = 0

    return (Experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. A function per arrival source

    The first approach we will use is creating an arrival generator per source.  There will be some code redundancy, but it will a clear design for others to understand.
    """)
    return


@app.cell
def _(itertools):
    def shoulder_arrivals_generator(env, trace_enabled, args):
        """
        Arrival process for shoulders.

        Parameters:
        ------
        env: simpy.Environment
            The simpy environment for the simulation

        args: Experiment
            The settings and input parameters for the simulation.
        """
        # use itertools as it provides an infinite loop
        # with a counter variable that we can use for unique Ids
        for patient_count in itertools.count(start=1):
            # the sample distribution is defined by the experiment.
            inter_arrival_time = args.arrival_shoulder.sample()
            yield env.timeout(inter_arrival_time)

            args.results["n_shoulders"] = patient_count

            trace(f"{env.now:.2f}: SHOULDER arrival.", trace_enabled)

    return (shoulder_arrivals_generator,)


@app.cell
def _(itertools):
    def hip_arrivals_generator(env, trace_enabled, args):
        """
        Arrival process for hips.

        Parameters:
        ------
        env: simpy.Environment
            The simpy environment for the simulation

        args: Experiment
            The settings and input parameters for the simulation.
        """
        # use itertools as it provides an infinite loop
        # with a counter variable that we can use for unique Ids
        for patient_count in itertools.count(start=1):
            # the sample distribution is defined by the experiment.
            inter_arrival_time = args.arrival_hip.sample()
            yield env.timeout(inter_arrival_time)

            args.results["n_hips"] = patient_count
            trace(f"{env.now:.2f}: HIP arrival.", trace_enabled=trace_enabled)

    return (hip_arrivals_generator,)


@app.cell
def _(itertools):
    def wrist_arrivals_generator(env, trace_enabled, args):
        """
        Arrival process for wrists.

        Parameters:
        ------
        env: simpy.Environment
            The simpy environment for the simulation

        args: Experiment
            The settings and input parameters for the simulation.
        """
        # use itertools as it provides an infinite loop
        # with a counter variable that we can use for unique Ids
        for patient_count in itertools.count(start=1):
            # the sample distribution is defined by the experiment.
            inter_arrival_time = args.arrival_wrist.sample()
            yield env.timeout(inter_arrival_time)

            args.results["n_wrists"] = patient_count
            trace(f"{env.now:.2f}: WRIST arrival.", trace_enabled=trace_enabled)

    return (wrist_arrivals_generator,)


@app.cell
def _(itertools):
    def ankle_arrivals_generator(env, trace_enabled, args):
        """
        Arrival process for ankles.

        Parameters:
        ------
        env: simpy.Environment
            The simpy environment for the simulation

        args: Experiment
            The settings and input parameters for the simulation.
        """
        # use itertools as it provides an infinite loop
        # with a counter variable that we can use for unique Ids
        for patient_count in itertools.count(start=1):
            # the sample distribution is defined by the experiment.
            inter_arrival_time = args.arrival_wrist.sample()
            yield env.timeout(inter_arrival_time)

            args.results["n_ankles"] = patient_count
            trace(f"{env.now:.2f}: ANKLE arrival.", trace_enabled=trace_enabled)

    return (ankle_arrivals_generator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6. Single run function
    """)
    return


@app.cell
def _(
    RUN_LENGTH,
    ankle_arrivals_generator,
    hip_arrivals_generator,
    shoulder_arrivals_generator,
    simpy,
    wrist_arrivals_generator,
):
    def single_run(experiment, rep=0, run_length=RUN_LENGTH, trace_enabled=False):
        """
        Perform a single run of the model and return the results

        Parameters:
        -----------

        experiment: Experiment
            The experiment/paramaters to use with model

        rep: int
            The replication number.

        rc_period: float, optional (default=RUN_LENGTH)
            The run length of the model
        """

        # reset all results variables to zero and empty
        experiment.init_results_variables()

        # set random number set to the replication no.
        # this controls sampling for the run.
        experiment.set_random_no_set(rep)

        # environment is (re)created inside single run
        env = simpy.Environment()

        # we pass all arrival generators to simpy
        env.process(shoulder_arrivals_generator(env, trace_enabled, experiment))
        env.process(hip_arrivals_generator(env, trace_enabled, experiment))
        env.process(wrist_arrivals_generator(env, trace_enabled, experiment))
        env.process(ankle_arrivals_generator(env, trace_enabled, experiment))

        # run for warm-up + results collection period
        env.run(until=run_length)

        # return the count of the arrivals
        return experiment.results

    return (single_run,)


@app.cell
def _(Experiment, single_run):
    experiment = Experiment()
    results = single_run(experiment, trace_enabled=True)
    results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A hospital that only provides surgery for hip fractures
    """)
    return


@app.cell
def _(Experiment, single_run):
    M = 1_000_000
    experiment_2 = Experiment(iat_shoulder=M, iat_wrist=M, iat_ankle=M)
    results_2 = single_run(experiment_2)
    results_2
    return


if __name__ == "__main__":
    app.run()
