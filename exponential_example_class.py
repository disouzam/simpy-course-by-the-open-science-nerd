"""
Example of a class to encapsulate random number generation and sampling

Code by The Open Science Nerd at https://github.com/pythonhealthdatascience/intro-open-sim/blob/main/content/01_sampling.ipynb
"""

import numpy as np


class Exponential:
    """
    Convenience class for the exponential distribution.
    Packages up distribution parameters, seed and random generator.
    """

    def __init__(self, mean, random_seed=None):
        """
        Constructor

        Params:
        ------
        mean: float
            The mean of the exponential distribution

        random_seed: int | SeedSequence, optional (default=None)
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
        """
        return self.rand.exponential(self.mean, size=size)
