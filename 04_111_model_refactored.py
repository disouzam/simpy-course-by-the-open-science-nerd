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
    # A full SimPy model

    In this notebook, we will now build a full `simpy` process model. Our example is a queuing model of a 111 call centre.  We will include random arrivals and resources. We will keep this simple, and gradually add in detail and flexibility to our design.
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
    from collections import namedtuple

    import simpy

    return (simpy, namedtuple)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Problem background

    Call operators in an 111 (urgent care) service receive calls at a rate of 100 per hour. Call length can be represented by a triangular distribution.  Calls last between 5 minutes and 15 minutes. Most calls last 7 minutes. There are 13 call operators.


    ![Call Centre Diagram](public/callcentre.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. SimPy resources

    To model the call centre, we need to introduce a **resource**.

    These resources represent the call operators. If a resource is not available then a process will **pause** (i.e. callers will wait for an operator to become available).

    We create a resource as follows:

    ```python
    operators = simpy.Resource(env, capacity=20)
    ```

    When we want to request a resource in our process, we create a `with` block as follows:

    ```python
    with operators.request() as req:
        yield req
    ```

    This tells SimPy that **your process needs an operator resource to progress**.  The code will pause until a resource is yielded. This gives us our **queuing effect**.  If a resource is not available immediately then the process will wait until one becomes available.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The service function

    We will first create a python function called `service()` to simulate the service process for a call operator. We need to include the following logic:

    1. **Request and wait** (if necessary) for a call operator.
    2. Undergo **phone triage** (a delay). This is a sample from the Triangular distribution.
    3. **Exit the system**.

    Each caller that arrives in the simulation will this function as a SimPy **process**. As inputs to the function, we will pass:

    * A unique patient identifier (`identifier`)
    * A pool of operator resources (`operators`)
    * The environment (`env`)
    * The service process random number generator object (`service_rng`)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The generator function

    The generator function is very similar to the [our initial
    code](./03b_exercise1_solutions.py).

    > ✂️ **Notice the pattern**. For most models you can just cut, paste and modify code you have used before.
    """)
    return


@app.cell
def _():
    from arrivals_generator import arrivals_generator

    return (arrivals_generator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Run the model
    """)
    return


@app.cell
def _(arrivals_generator, simpy, namedtuple):
    import numpy as np

    # model parameters
    RUN_LENGTH = 100
    N_OPERATORS = 13

    # create simpy environment and operator resources
    env = simpy.Environment()
    operators = simpy.Resource(env, capacity=N_OPERATORS)

    results_dict = {}
    results_dict["waiting_times"] = []
    results_dict["call_durations"] = []

    # total operator usage time for utilisation calculation.
    results_dict["total_call_duration"] = 0.0

    args_tuple = namedtuple(
        "args_tuple", ["operators", "arrival_dist", "call_dist", "results"]
    )

    # create the arrival process rng
    arrival_rng = np.random.default_rng()

    # create the service rng that we pass to each service process created
    service_rng = np.random.default_rng()

    class Arrival_dist:
        def __init__(self, rng):
            self.rng = rng

        def sample(self):
            return self.rng.exponential(60 / 100)

    class Call_dist:
        def __init__(self, rng):
            self.rng = rng

        def sample(self):
            return self.rng.triangular(left=5.0, mode=7.0, right=10.0)

    arrival_dist = Arrival_dist(arrival_rng)
    call_dist = Call_dist(service_rng)
    args = args_tuple(
        operators=operators,
        arrival_dist=arrival_dist,
        call_dist=call_dist,
        results=results_dict,
    )

    env.process(arrivals_generator(env, args, operators))
    env.run(until=RUN_LENGTH)
    print(f"end of run. simulation clock time = {env.now}")
    return


if __name__ == "__main__":
    app.run()
