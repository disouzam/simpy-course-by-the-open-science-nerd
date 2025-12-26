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
    # Generator exercise

    🧐 For the solutions, please see the [generator exercise solutions notebook](./03b_exercise1_solutions.ipynb)
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
    import numpy as np
    import simpy

    return np, simpy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example code

    The code below is taken from the simple call centre example.  In this code arrivals occur with an inter-arrival time (IAT) of exactly 1 minute.
    """)
    return


@app.function
def arrivals_generator(env):
    """
    Prescriptions arrive with a fixed duration of 1 minute.

    Parameters:
    ------
    env: simpy.Environment
    """

    # don't worry about the infinite while loop, simpy will
    # exit at the correct time.
    while True:
        # sample an inter-arrival time.
        inter_arrival_time = 1.0

        # we use the yield keyword instead of return
        yield env.timeout(inter_arrival_time)

        # print out the time of the arrival
        print(f"Call arrives at: {env.now}")


@app.cell
def _(simpy):
    # model parameters
    RUN_LENGTH = 25

    # create the simpy environment object
    env = simpy.Environment()

    # tell simpy that the `arrivals_generator` is a process
    env.process(arrivals_generator(env))

    # run the simulation model
    env.run(until=RUN_LENGTH)
    print(f"end of run. simulation clock time = {env.now}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Exercise: Modelling a poisson arrival process for prescriptions

    **Task:**

    Update `arrivals_generator()` so that inter-arrival times follow an **exponential distribution** with a mean inter-arrival time of 60.0 / 100 minutes between arrivals (i.e. 100 arrivals per hour). Use a run length of 25 minutes.

    **Bonus challenge:**

    * First, try implementing this **without** setting a random seed.
    * Then, update the method with an approach to control the randomness,

    **Hints:**

    * We learnt how to sample using a `numpy` random number generator in the [sampling notebook](./01_sampling.ipynb). Excluding a random seed, the basic method for drawing a single sample follows this pattern:
        ```python
        rng = np.random.default_rng()
        sample = rng.exponential(scale=12.0)
        ```
    """)
    return


@app.cell
def _(np):
    def arrivals_generator_exponential(env):
        """
        Prescriptions arrive with a exponential arrival time.

        Parameters:
        ------
        env: simpy.Environment
        """
        rng = np.random.default_rng(seed=1)

        # don't worry about the infinite while loop, simpy will
        # exit at the correct time.
        while True:
            # sample an inter-arrival time.
            inter_arrival_time = rng.exponential(scale=60 / 100)

            # we use the yield keyword instead of return
            yield env.timeout(inter_arrival_time)

            # print out the time of the arrival
            print(f"Call arrives at: {env.now:.3f}")

    return (arrivals_generator_exponential,)


@app.cell
def _(arrivals_generator_exponential, simpy):
    # model parameters
    RUN_LENGTH_EXPONENTIAL = 25

    # create the simpy environment object
    env_exponential = simpy.Environment()

    # tell simpy that the `arrivals_generator` is a process
    env_exponential.process(arrivals_generator_exponential(env_exponential))

    # run the simulation model
    env_exponential.run(until=RUN_LENGTH_EXPONENTIAL)
    print(f"end of run. simulation clock time = {env_exponential.now:.3f}")
    return


if __name__ == "__main__":
    app.run()
