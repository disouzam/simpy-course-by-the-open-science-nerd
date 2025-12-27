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
    # Collecting results from a single run

    When running our DES, we want to **collect data** that helps us to analyse the system performance, such as **wait times, resource utilisation, and queue lengths**.

    A tool like `simpy` allows you to collect your data flexibly using an approach that makes sense to you! Some options are:

    1. **Code an auditor / observer process**.  This process will periodically observe the state of the system. We can use this to collect information on **current state at time t**. For example, how many patients are queuing and how many have a call in progress between by time of day.

    2. **Store process metrics during a run and perform calculations at the end of a run**. For example, if you want to calculate mean patient waiting time then store each patient waiting time in a list and calculate the mean at the end of the run.

    3. **Conduct and audit or calculate running statistics as the simulation executes an event**.  For example, as a patient completes a call we can calculate a running mean of waiting times and a running total of the operators are taking calls. The latter measure can then be used to calculate server utilisation. You could also use this approach to audit queue length where the queue length is recorded each time request for a resource is made (and/or when a resource is released).

    This notebook provides an example of the **second strategy**.
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
    import itertools

    import numpy as np
    import simpy

    from logging_and_tracing import trace

    return itertools, np, simpy, trace


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Calculating mean waiting time

    The second strategy to results collection is to store either a reference to a quantitative value (e.g. waiting time) during the run.  Once the run is complete you will need to include a procedure for computing the metric of interest.

    * 😊 An advantage of this strategy is that it is very **simple**, captures all data, and has minimal computational overhead during a model run!
    * 😢 A potential disadvantage is that for complex simulation you may end up storing a **large amount of data in memory**. In these circumstances, it may be worth exploring event driven strategies to reduce memory requirements.

    ![](./public/callcentre_waittime.png)

    In our example, we will:

    1. Create a **list** (`results['waiting_times']`) to store each caller's wait time
    2. When the model runs, **each time a caller enters service**, the `service()` function will append a `waiting_time` for the caller to the list
    3. At the end of the run, we will loop through these references and calculate **mean waiting time**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 Service and arrival functions

    The only modification we need to make is to the `service` function.  We will add in a line of code to record the `waiting_time` of the caller as they enter service.

    ```python
    results['waiting_times'].append(waiting_time)
    ```
    """)
    return


@app.cell
def _(trace):
    def service(identifier, operators, env, service_rng, results_dict, trace_enabled):
        """
        Simulates the service process for a call operator

        1. request and wait for a call operator
        2. phone triage (triangular)
        3. exit system

        Params:
        ------

        identifier: int
            A unique identifer for this caller

        operators: simpy.Resource
            The pool of call operators that answer calls
            These are shared across resources.

        env: simpy.Environment
            The current environent the simulation is running in
            We use this to pause and restart the process after a delay.

        service_rng: numpy.random.Generator
            The random number generator used to sample service times

        """
        # record the time that call entered the queue
        start_wait = env.now

        # request an operator
        with operators.request() as req:
            yield req

            # record the waiting time for call to be answered
            waiting_time = env.now - start_wait
            results_dict["waiting_times"].append(waiting_time)

            trace(f"operator answered call {identifier} at " + f"{env.now:.3f}")

            # sample call duration.
            call_duration = service_rng.triangular(left=5.0, mode=7.0, right=10.0)

            # schedule process to begin again after call_duration
            yield env.timeout(call_duration)

            # print out information for patient.
            trace(
                f"call {identifier} ended {env.now:.3f}; "
                + f"waiting time was {waiting_time:.3f}",
                trace_enabled,
            )

    return (service,)


@app.cell
def _(itertools, np, service, trace):
    def arrivals_generator(env, operators, results_dict, trace_enabled):
        """
        IAT is exponentially distributed

        Parameters:
        ------
        env: simpy.Environment
            The simpy environment for the simulation

        operators: simpy.Resource
            the pool of call operators.
        """
        # create the arrival process rng
        arrival_rng = np.random.default_rng()

        # create the service rng that we pass to each service process created
        service_rng = np.random.default_rng()

        # use itertools as it provides an infinite loop
        # with a counter variable that we can use for unique Ids
        for caller_count in itertools.count(start=1):
            # 100 calls per hour (sim time units = minutes).
            inter_arrival_time = arrival_rng.exponential(60 / 100)
            yield env.timeout(inter_arrival_time)

            trace(f"call arrives at: {env.now:.3f}", trace_enabled)

            # create a new simpy process for serving this caller.
            # we pass in the caller id, the operator resources, env, and the rng
            env.process(
                service(
                    caller_count,
                    operators,
                    env,
                    service_rng,
                    results_dict,
                    trace_enabled,
                )
            )

    return (arrivals_generator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.4 Conduct a single run of the model

    We could keep the code to run the model as a script. However, it is useful to create a new **function** called `single_run` that we use to perform a **single replication** of the model and return results.

    If we later want to run **multiple replications** it is just a case of running `single_run` in a **loop**.

    We add a line of code to find the mean waiting times from `results`.
    """)
    return


@app.cell
def _(arrivals_generator, simpy):
    def single_run(run_length, n_operators, results_dict, trace_enabled):
        """
        Perform a single replication of the simulation model and
        return the mean waiting time as a result.

        Parameters:
        ----------
        run_length: float
            The duration of the simulation run in minutes.

        n_operators: int
            The number of call operators to create as a resource

        Returns:
        -------
        mean_waiting_time: int
        """
        # create simpy environment and operator resources
        env = simpy.Environment()
        operators = simpy.Resource(env, capacity=n_operators)

        env.process(arrivals_generator(env, operators, results_dict, trace_enabled))
        env.run(until=run_length)
        print(f"end of run. simulation clock time = {env.now}")

        return results_dict

    return (single_run,)


@app.cell
def _(np, single_run):
    # reset data structure holding results
    results = {}
    results["waiting_times"] = []

    # model parameters
    RUN_LENGTH = 1000
    N_OPERATORS = 13

    # Turn off caller level results.
    trace_enabled = True

    single_run(RUN_LENGTH, N_OPERATORS, results, trace_enabled)
    mean_waiting_time = np.mean(results["waiting_times"])
    print("Simulation Complete")
    print(f"Waiting time for call operators: {mean_waiting_time:.2f} minutes")
    return


if __name__ == "__main__":
    app.run()
