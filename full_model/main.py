# from callcentresim.model import Experiment, multiple_replications
from execution import multiple_replications

from experiment import Experiment

# experiment parameters

# no. resources
n_operators = 13
n_nurses = 9

# demand
mean_iat = 0.6

# patient routing
chance_call_back = 0.4

# set number of replications
n_reps = 5

# set warm-up period
warm_up_period = 100.0
results_collection_period = 1000.0

user_experiment = Experiment(
    n_operators=n_operators,
    n_nurses=n_nurses,
    mean_iat=mean_iat,
    chance_callback=chance_call_back,
)

results = multiple_replications(
    experiment=user_experiment,
    wu_period=warm_up_period,
    rc_period=results_collection_period,
    n_reps=n_reps,
)

print(results.describe().round(2).T)
