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
    # Experiments 🧪

    A key part of any computer simulation study is **experimentation**.  Here a set of experiments will be conducted in an attempt to understand and find improvements to the system under study.  Experiments essentially vary inputs and process logic.

    We can do this manually, but as we develop a model the number of input parameters will increase.

    💡 There are several data structures you might employ to organise parameters.

    * a python dictionary
    * a custom parameter class
    * a dataclass

    All of these approaches work well and it really is a matter of judgement on what you prefer. One downside of a python `dict` and a custom class is that they are both mutable (although a class can have custom properties where users can only access 'viewable' attributes).  A dataclass can easily be made immutable and requires less code than a custom class, but has the downside that its syntax is a little less pythonic. Here we will build a parameter class called `Experiment`.

    > ☺️ We will also use this re-organisation of code to eliminate our global variables!
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
    import pandas as pd
    import simpy

    import single_run as sr
    from sensible_constants import RESULTS_COLLECTION_PERIOD, TRACE

    return RESULTS_COLLECTION_PERIOD, TRACE, np, pd, simpy, sr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Notebook level variables, constants, and default values

    A useful first step when setting up a simulation model is to define the base case or as-is parameters.  Here we will create a set of constant/default values for our `Experiment` class, but you could also consider reading these in from a file.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Distribution classes

    We will define two distribution classes (`Triangular` and `Exponential`) to encapsulate the random number generation, parameters and random seeds used in the sampling.  This simplifies what we will need to include in the `Experiment` class and as we will see later makes it easier to vary distributions as well as parameters.
    """)
    return


@app.cell
def _():
    # import distributions as dist
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Experiment class

    An experiment class is useful because it allows use to easily configure and schedule a large number of experiments to occur in a loop.  We set the class up so that it uses the default variables we defined above i.e. as default the model reflects the as-is process.  To run a new experiment we simply override the default values.
    """)
    return


@app.cell
def _():
    from experiment import Experiment

    return (Experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1. Creating a default experiment
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To use `Experiment` is very simple.  For example to create a default experiment (that uses all the default parameter values) we would use the following code
    """)
    return


@app.cell
def _(Experiment, simpy):
    env = simpy.Environment()
    _ = env
    default_experiment = Experiment()
    return (default_experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Due to the way python works we can access all of the experiment variables from the `default_scenario` object. For example the following code will generate an inter-arrival time:
    """)
    return


@app.cell
def _(default_experiment):
    default_experiment.arrival_dist.sample()
    return


@app.cell
def _(default_experiment):
    default_experiment.mean_iat
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Creating an experiment with more call operators

    To change parameters in an experiment we just need to include a new value when we create the `Experiment`.  For example if we wanted to increase the number of servers to 14. We use the following code:
    """)
    return


@app.cell
def _(Experiment, simpy):
    env_2 = simpy.Environment()
    _ = env_2
    extra_server = Experiment(n_operators=14)
    return (extra_server,)


@app.cell
def _(extra_server):
    extra_server.n_operators
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Modified model code

    We will modify the model code and logic that we have already developed.  The functions for service and arrivals will now accept an `Experiment` argument.

    > Note that at this point you could put all of the code into a python module and import the functions and classes you need into an experiment workbook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. A single run wrapper function
    """)
    return


@app.cell
def _(Experiment, TRACE, sr):
    default_scenario = Experiment()
    results = sr.single_run(default_scenario, TRACE)
    return (results,)


@app.cell
def _(results):
    print(
        f"Mean waiting time: {results['01_mean_waiting_time']:.2f} mins \n"
        f"Mean call duration: {results['04_mean_call_duration']:.2f} mins \n"
        + f"Operator Utilisation {results['02_operator_util']:.2f}%\n"
        + f"Total call duration {results['03_total_call_duration']:.2f} mins"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multiple Replications
    """)
    return


@app.cell
def _(RESULTS_COLLECTION_PERIOD, np, pd, sr):
    def multiple_replications(
        experiment, rc_period=RESULTS_COLLECTION_PERIOD, n_reps=5
    ):
        """
        Perform multiple replications of the model.

        Params:
        ------
        experiment: Experiment
            The experiment/paramaters to use with model

        rc_period: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
            results collection period.
            the number of minutes to run the model to collect results

        n_reps: int, optional (default=5)
            Number of independent replications to run.

        Returns:
        --------
        pandas.DataFrame
        """

        # loop over single run to generate results dicts in a python list.
        results = [sr.single_run(experiment, rep, rc_period) for rep in range(n_reps)]

        # format and return results in a dataframe
        df_results = pd.DataFrame(results)
        df_results.index = np.arange(1, len(df_results) + 1)
        df_results.index.name = "rep"
        return df_results

    return (multiple_replications,)


@app.cell
def _(Experiment, mo, multiple_replications):
    default_scenario_2 = Experiment()
    results_2 = multiple_replications(default_scenario_2)

    format_mapping = {}
    for col in results_2.columns:
        format_mapping[col] = "{:.2f}"

    mo.ui.table(results_2, format_mapping=format_mapping)
    return


if __name__ == "__main__":
    app.run()
