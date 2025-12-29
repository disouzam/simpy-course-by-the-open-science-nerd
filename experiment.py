import numpy as np

import distributions as dist
import sensible_constants as sconst


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
        random_number_set=sconst.DEFAULT_RND_SET,
        n_operators=sconst.N_OPERATORS,
        mean_iat=sconst.MEAN_IAT,
        call_low=sconst.CALL_LOW,
        call_mode=sconst.CALL_MODE,
        call_high=sconst.CALL_HIGH,
        n_streams=sconst.N_STREAMS,
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
        self.arrival_dist = dist.Exponential(self.mean_iat, random_seed=self.seeds[0])

        # duration of call triage
        self.call_dist = dist.Triangular(
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
        self.results["call_durations"] = []

        # total operator usage time for utilisation calculation.
        self.results["total_call_duration"] = 0.0
