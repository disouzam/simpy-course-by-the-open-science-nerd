import numpy as np


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
