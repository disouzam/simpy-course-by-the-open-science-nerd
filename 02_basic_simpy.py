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
    # An introduction to SimPy

    In this tutorial, we will use **SimPy**, a free and open-source software (FOSS) framework for discrete-event simulation.

    ## 1. Why choose FOSS and SimPy?

    💪 A strength of SimPy is its **simplicity and flexibility**.

    * As it is part of Python, it is often straightforward to use SimPy to model complex logic and make use of the [SciPy stack](https://projects.scipy.org/stackspec.html)!

    📝 You will initially need to **write lots of code** - or borrow code from existing simulation studies online. Do not worry though! As you use SimPy, you will build up your own library of reusable code that you can draw upon and build on for future simulation projects.

    ♻️ SimPy is **FOSS** - the benefits of this for research are that:

    * Model logic is **transparent**
    * It can be readily **shared** with others
    * It can easily **link to other data science tools** (e.g. `sklearn` or `pytorch` for machine learning)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Imports

    For `simpy`, the typical style is to import the whole package as follows:
    """)
    return


@app.cell
def _():
    import simpy

    return (simpy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. An example: a urgent care call sample

    This case study uses a simple model of an urgent care telephone call centre, similar to the NHS 111 service in the UK.  To learn `simpy` we will first build a very simple model. In our first iteration of this model, calls to the centre arrive **deterministically**.  For now we will ignore resources and activities in the model and just model a deterministic arrival process.   The simulation time units are in minutes.  Let's assume there are 60 new callers per hour (an fixed inter-arrival time of 1.0 per minute).

    ## 4. The model building blocks

    To build our model, we will need the following components...

    ### 4.1 A SimPy environment

    `simpy` has process based worldview.  These processes take place in an environment.  You can create a environment with the following line of code:

    ```python
    env = simpy.Environment()
    ```

    ### 4.2 SimPy timeouts

    We can introduce **delays** or **activities** into a process.  For example these might be the duration of a stay on a ward, or the duration of a operation - or, in this case, a **delay between arrivals (inter-arrival time)**. In `simpy` you control this with the following method:

    ```python
    env.timeout(1.0)
    ```

    ### 4.3 Generators

    The events in the DES are modelled and scheduled in `simpy` using python **generators** (i.e. they are the "event-processing mechanism"). A generator is a function that behaves like an iterator, meaning it can yield a **sequence of values** when iterated over.

    For example, below is a basic generator function that yields a new arrival every 1 minute. It takes the **environment** as a parameter. It then internally calls the `env.timeout()` method in an infinite loop.

    ```python
    def arrivals_generator(env):
        while True:
            yield env.timeout(1.0)
    ```

    ### 4.4 SimPy process and run

    Once we have coded the model logic and created an environment instance, there are two remaining instructions we need to code.

    1. Set the generator up as a **SimPy process** using `env.process()`

    ```python
    env.process(arrivals_generator(env))
    ```

    2. Run the environment for a user specified **run length** using `env.run()`

    ```python
    env.run(until=25)
    ```

    The run method handle the infinite loop we set up in `arrivals_generator`. The simulation model has an internal concept of time.  It will end execution when its internal clock reaches 25 time units.

    ## 5. Create the model

    **Now that we have covered the basic building blocks, let's code the actual model.**  It makes sense to create our model logic first.  The code below will generate arrivals every 60.0 / 100.0 minutes.  Note that the function takes an environment object as a parameter.
    """)
    return


@app.function
def arrivals_generator(env):
    """
    Callers arrive with a fixed inter-arrival time of 1.0 minutes.

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have our generator function we can setup the environment, process and call run.  We will create a `RUN_LENGTH` parameter that you can change to run the model for different time lengths.

    **Consider:** What would happen if we set `RUN_LENGTH` to 50?
    """)
    return


@app.cell
def _(simpy):
    # model parameters
    RUN_LENGTH = 50

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
    ## 6. Exercise

    Before we learn anything more about `simpy`, have a go at the [generators exercise](./03a_exercise1.ipynb).

    In the exercise you will need to modify the `arrivals_generator` so that it has random arrivals. This exercise tests that you have understood the basics of `simpy` and random sampling in `numpy`
    """)
    return


if __name__ == "__main__":
    app.run()
