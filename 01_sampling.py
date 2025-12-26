import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
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
    # Sampling from statistical distributions in Python

    If you are working in simulation modelling in Python, you will likely need to use `numpy.random` namespace. It provides a variety of statistical distributions which you can use for efficient sampling.

    This notebook will guide you through examples of

    1.  Creating instances of a high quality Pseudo Random Number Generator (PRNG) using PCG64 provided by `numpy`
    2.  Generating samples from the **uniform**, **exponential** and **normal** distributions.
    3.  Spawning multiple non-overlapping streams of random numbers
    4.  Using OOP to encapsulate PRNGs, distributions and parameters for simulation models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports

    We will import `numpy` for our sampling and `matplotlib` to plot our distributions.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Helper functions

    The simple function below can be used to automatically produce a plot illustrating a distribution of samples.
    """)
    return


@app.cell
def _(np, plt):
    def distribution_plot(samples, bins=100, figsize=(5, 3)):
        """
        Helper function to visualise the distributions

        Params:
        -----
        samples: np.ndarray
            A numpy array of quantitative data to plot as a histogram.

        bins: int, optional (default=100)
            The number of bins to include in the histogram

        figsize: (int, int) (default=(5,3))
            Size of the plot in pixels

        Returns:
        -------
            fig, ax: a tuple containing matplotlib figure and axis objects.
        """
        hist = np.histogram(samples, bins=np.arange(bins), density=True)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        _ = ax.plot(hist[0])
        _ = ax.set_ylabel("p(x)")
        _ = ax.set_xlabel("x")

        return fig, ax

    return (distribution_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Creating a random number generator object

    To generate pseudo random numbers for sampling from each distribution, we can use the `default_rng()` function from the `numpy.random` module.

    This function constructs an instance of a `Generator` class, which can produce random numbers.

    By default `numpy` uses a Pseudo-Random Number Generator (PRNG) called use of the [Permuted Congruential Generator 64-bit](https://www.pcg-random.org/) (PCG64; period = $2^{128}$; maximum number of streams = $2^{127}$).

    For more information on `Generator` you can look at [`numpy` online documentation.](https://numpy.org/doc/stable/reference/random/generator.html)
    """)
    return


@app.cell
def _(np):
    rng = np.random.default_rng()
    type(rng)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Steps to create a sample

    In general, the approach to sampling is:

    1. Create a random number **generator** object

    2. Using the object call the method for the **statistical distribution**
        * Each method has its own custom parameters
        * Each method will include a `size` parameter that you use to set the number of samples to generate

    3. **Store** the result in an appropriately named variable

    ### 4.1 Uniform distribution
    """)
    return


@app.cell
def _(distribution_plot, np, plt):
    # Step 1: create a random number generator object - set seed to 42
    rng_1 = np.random.default_rng()

    # Step 2 and 3: call the appropriate method of the generator and store result
    samples_1 = rng_1.uniform(low=10, high=40, size=1_000_000)

    # Illustrate with plot.
    fig_1, ax_1 = distribution_plot(samples_1, bins=50)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Exponential distribution
    """)
    return


@app.cell
def _(distribution_plot, np, plt):
    rng_2 = np.random.default_rng(42)
    samples_2 = rng_2.exponential(scale=12, size=1_000_000)
    fig_2, ax_2 = distribution_plot(samples_2, bins=50)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.3 Normal distribution
    """)
    return


@app.cell
def _(distribution_plot, np, plt):
    rng_3 = np.random.default_rng(42)
    samples_3 = rng_3.normal(loc=25.0, scale=5.0, size=1_000_000)
    fig_3, ax_3 = distribution_plot(samples_3, bins=50)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.4 Generating a single sample

    If we just need to generate the a single sample we omit the `size` parameter. This returns a scalar value.
    """)
    return


@app.cell
def _(np):
    rng_4 = np.random.default_rng(42)
    sample_4 = rng_4.normal(loc=25.0, scale=5.0)
    print(sample_4)
    print(type(sample_4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note** that you can also set `size` to 1.  Just be aware that an array is returned. e.g.
    """)
    return


@app.cell
def _(np):
    rng_5 = np.random.default_rng(42)
    sample_5 = rng_5.normal(loc=25.0, scale=5.0, size=1)
    # a numpy array is returned
    print(sample_5)
    print(type(sample_5))

    # to access the scalar value use the 0 index of the array.
    print(sample_5[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Spawning multiple non-overlapping PRN streams.

    For simulation we ideally want to use multiple streams of random numbers that do not overlap (i.e. they are independent). This is straightforward to implement in Python using `SeedSequence` and a user provided integer seed and the number of independent streams to spawn.

    > As a user we don't need to worry about the quality of the integer seed provided. This is useful for implementing multiple replications and common random numbers.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Here's how we create the seeds from a single user supplied seed.  The returned variable `seeds` is a Python `List`.
    """)
    return


@app.cell
def _(np):
    n_streams = 2
    user_seed = 1

    seed_sequence = np.random.SeedSequence(user_seed)
    seeds = seed_sequence.spawn(n_streams)
    return (seeds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use `seeds` when creating our PRNGs.  For example, one for inter-arrival times and one for service times.
    """)
    return


@app.cell
def _(np, seeds):
    # e.g. to model arrival times
    arrival_rng = np.random.default_rng(seeds[0])
    arrival_sample = arrival_rng.normal(loc=25.0, scale=5.0)
    print(arrival_sample)

    # e.g. to model service times
    service_rng = np.random.default_rng(seeds[1])
    service_sample = service_rng.normal(loc=25.0, scale=5.0)
    print(service_sample)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Encapsulating distributions, parameters, and random seeds.

    When building a simulation model it is often useful to *package up* both a random number generator, parameters for a specific distribution, and a seed in a **python class**.  This allows **easy creation** of generator objects, straightforward sampling, and improves management of streams for each activity in a simulation model.

    As an example below, the class `Exponential` represents the exponential distribution. It accepts a mean value parameter and you can set the random seed.

    We will then instantiate two `Exponential` objects for two different processes in our simulation: acute length of stay, and rehab length of stay.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    As opposed to the original code, I preferred to move definition of Exponential class to a separate file (_exponential_example_class.py_) to demonstrate the value of modularization of code. It is also interesting to keep notebook as short as possible.
    """)
    return


@app.cell
def _():
    from exponential_example_class import Exponential

    return (Exponential,)


@app.cell
def _(Exponential):
    acute_los = Exponential(3.0, random_seed=42)
    rehab_los = Exponential(30.0, random_seed=101)
    return acute_los, rehab_los


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First samples
    """)
    return


@app.cell
def _(acute_los):
    acute_los.sample()
    return


@app.cell
def _(rehab_los):
    rehab_los.sample()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Second samples
    """)
    return


@app.cell
def _(acute_los):
    acute_los.sample()
    return


@app.cell
def _(rehab_los):
    rehab_los.sample()
    return


if __name__ == "__main__":
    app.run()
