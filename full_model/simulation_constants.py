EPSILON = 1e-3

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
TRACE = True

# run variables
RESULTS_COLLECTION_PERIOD = 1000

# ##############################################################################
# MODIFICATON: added a warm-up period, by default we will not use it.
WARM_UP_PERIOD = 0
# ##############################################################################
