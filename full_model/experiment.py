import numpy as np
import simulation_constants as sim_const
from probability_distributions import Bernoulli, Exponential, Triangular, Uniform


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
        random_number_set=sim_const.DEFAULT_RND_SET,
        n_streams=sim_const.N_STREAMS,
        n_operators=sim_const.N_OPERATORS,
        mean_iat=sim_const.MEAN_IAT,
        call_low=sim_const.CALL_LOW,
        call_mode=sim_const.CALL_MODE,
        call_high=sim_const.CALL_HIGH,
        # ######################################################################
        # MODIFICATION: nurse parameters
        n_nurses=sim_const.N_NURSES,
        chance_callback=sim_const.CHANCE_CALLBACK,
        nurse_call_low=sim_const.NURSE_CALL_LOW,
        nurse_call_high=sim_const.NURSE_CALL_HIGH,
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

        # create the callback and nurse consultation distributions
        self.callback_dist = Bernoulli(self.chance_callback, random_seed=self.seeds[2])

        self.nurse_dist = Uniform(
            self.nurse_call_low,
            self.nurse_call_high,
            random_seed=self.seeds[3],
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

        # nurse sub process results collection
        self.results["nurse_waiting_times"] = []
        self.results["total_nurse_call_duration"] = 0.0


def create_experiments(df_experiments):
    """
    Returns dictionary of Experiment objects based on contents of a dataframe

    Params:
    ------
    df_experiments: pandas.DataFrame
        Dataframe of experiments. First two columns are id, name followed by
        variable names.  No fixed width

    Returns:
    --------
    dict
    """
    experiments = {}

    # experiment input parameter dictionary
    exp_dict = df_experiments[df_experiments.columns[1:]].T.to_dict()
    # names of experiments
    exp_names = df_experiments[df_experiments.columns[0]].T.to_list()

    print(exp_dict)
    print(exp_names)

    # loop through params and create Experiment objects.
    for name, params in zip(exp_names, exp_dict.values()):
        print(name)
        experiments[name] = Experiment(**params)

    return experiments
