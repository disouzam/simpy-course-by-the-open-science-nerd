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
    # 1. Imports
    """)
    return


@app.cell
def _():
    import numpy as np

    return (np,)


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

    # Boolean switch to display simulation results as the model runs
    TRACE = False

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
        TRACE,
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
            self.arrival_shoulder = Exponential(
                self.iat_shoulder, random_seed=self.seeds[0]
            )

            self.arrival_hip = Exponential(self.iat_hip, random_seed=self.seeds[0])

            self.arrival_wrist = Exponential(self.iat_wrist, random_seed=self.seeds[0])

            self.arrival_ankle = Exponential(self.iat_ankle, random_seed=self.seeds[0])

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

    return


if __name__ == "__main__":
    app.run()
