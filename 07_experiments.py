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
    from sensible_constants import N_OPERATORS, RESULTS_COLLECTION_PERIOD, TRACE

    return N_OPERATORS, RESULTS_COLLECTION_PERIOD, TRACE, np, pd, simpy, sr


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Multiple experiments 🧪🧪🧪

    The `single_run` wrapper function for the model and the `Experiment` class mean that is very simple to run multiple experiments.  We will define two new functions for running multiple experiments:

    * `get_experiments()` - this will return a python dictionary containing a unique name for an experiment paired with an `Experiment` object
    * `run_all_experiments()` - this will loop through the dictionary, run all experiments and return combined results.
    * `experiment_summary_frame()` - take the results from each scenario and format into a simple table.
    """)
    return


@app.cell
def _(Experiment, N_OPERATORS):
    def get_experiments():
        """
        Creates a dictionary object containing
        objects of type `Experiment` 🧪 to run.

        Returns:
        --------
        dict
            Contains the experiments for the model
        """
        experiments = {}

        # base (default) case
        experiments["base"] = Experiment()

        # +1 extra capacity
        experiments["operators+1"] = Experiment(
            n_operators=N_OPERATORS + 1,
        )

        return experiments

    return (get_experiments,)


@app.cell
def _(RESULTS_COLLECTION_PERIOD, multiple_replications):
    def run_all_experiments(experiments, rc_period=RESULTS_COLLECTION_PERIOD):
        """
        Run each of the scenarios for a specified results
        collection period and replications.

        Params:
        ------
        experiments: dict
            dictionary of Experiment objects

        rc_period: float
            model run length

        """
        print("Model experiments:")
        print(f"No. experiments to execute = {len(experiments)}\n")

        experiment_results = {}
        for exp_name, experiment in experiments.items():
            print(f"Running {exp_name}", end=" => ")
            results = multiple_replications(experiment, rc_period)
            print("done.\n")

            # save the results
            experiment_results[exp_name] = results

        print("All experiments are complete.")

        # format the results
        return experiment_results

    return (run_all_experiments,)


@app.cell
def _(get_experiments, run_all_experiments):
    # get the experiments
    experiments = get_experiments()

    # run the scenario analysis
    experiment_results = run_all_experiments(experiments)
    return (experiment_results,)


@app.cell
def _(experiment_results):
    experiment_results["operators+1"]
    return


@app.cell
def _(pd):
    def experiment_summary_frame(experiment_results):
        """
        Mean results for each performance measure by experiment

        Parameters:
        ----------
        experiment_results: dict
            dictionary of replications.
            Key identifies the performance measure

        Returns:
        -------
        pd.DataFrame
        """
        columns = []
        summary = pd.DataFrame()
        for sc_name, replications in experiment_results.items():
            summary = pd.concat([summary, replications.mean()], axis=1)
            columns.append(sc_name)

        summary.columns = columns
        return summary

    return (experiment_summary_frame,)


@app.cell
def _(experiment_results, experiment_summary_frame):
    # as well as rounding you may want to rename the cols/rows to
    # more readable alternatives.
    summary_frame = experiment_summary_frame(experiment_results)
    summary_frame.round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load and multiple experiments from a CSV

    The `Experiment` class provides a simple way to run multiple experiments in a batch. To do so we can create multiple instances of Experiment, each with a different set of inputs for the model. These are then executed in a loop.

    ### Formatting experiment files

    In the format used here each row represents an experiment. The first column is a unique numeric identifier, the second column a name given to the experiment, and following $n$ columns represent the optional input variables that can be passed to an Experiment.

    Note that the method described here relies on the names of these columns matching the input parameters to `Experiment`.

       >  But note that columns do not need to be in the same order as Experiment arguments and they do not need to be exhaustive. A selection works fine.

    For example, in the urgent care call centre we will include 3 columns with the names:

    * n_operators
    * mean_iat

    The function `create_example_csv()` creates such a file containing four experiments that vary these paramters.
    """)
    return


@app.cell
def _(pd):
    def create_example_csv(filename="example_experiments.csv"):
        """
        Create an example CSV file to use in tutorial.
        This creates 4 experiments that varys
        n_operators, and mean_iat.

        Params:
        ------
        filename: str, optional (default='example_experiments.csv')
            The name and path to the CSV file.
        """
        # each column is defined as a seperate list
        names = ["base", "op+1", "high_demand", "combination"]
        operators = [13, 14, 13, 14]
        mean_iat = [0.6, 0.6, 0.55, 0.55]

        # empty dataframe
        df_experiments = pd.DataFrame()

        # create new columns from lists
        df_experiments["experiment"] = names
        df_experiments["n_operators"] = operators
        df_experiments["mean_iat"] = mean_iat

        df_experiments.to_csv(filename, index_label="id")

    return (create_example_csv,)


@app.cell
def _(create_example_csv, pd):
    create_example_csv()

    # load and illustrate results
    pd.read_csv("example_experiments.csv", index_col="id")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Converting the CSV to instances of `Experiment`

    The code above displays the experiments to the user and stores as a `pd.Dataframe` in the `df_experiments` variable. To convert the rows to `Experiment` objects is a two step process.

    * We cast the `Dataframe` to a nested python dictionary. Each key in the dictionary is the name of an experiment. The value is another dictionary where the key/value pairs are columns and their values.

    * We loop through the dictionary entries and pass the parameters to a new instance of the `Experiment` class.

    The function `create_experiments` implements both of these steps. The function returns a new dictionary where the key value pairs are the experiment name string, and an instance of `Experiment`
    """)
    return


@app.cell
def _(Experiment):
    def create_experiments(df_experiments):
        """
        Returns dictionary of Experiment objects based on contents of a dataframe

        Params:
        ------
        df_experiments: pandas.DataFrame
            Dataframe of experiments. First two columns are id, name followed by
            variable names.  No fixed width

        Returns:
        --------
        dict
        """
        experiments = {}

        # experiment input parameter dictionary
        exp_dict = df_experiments[df_experiments.columns[1:]].T.to_dict()
        # names of experiments
        exp_names = df_experiments[df_experiments.columns[0]].T.to_list()

        # loop through params and create Experiment objects.
        for name, params in zip(exp_names, exp_dict.values()):
            experiments[name] = Experiment(**params)

        return experiments

    return (create_experiments,)


@app.cell
def _(create_experiments, pd):
    # test of the function

    # assume code is run in same directory as example csv file
    df_experiment = pd.read_csv("example_experiments.csv", index_col="id")

    # convert to dict containing separate Experiment objects
    experiments_to_run = create_experiments(df_experiment)

    print(type(experiments_to_run))
    print(experiments_to_run["op+1"].n_operators)
    return (experiments_to_run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run all experiments and show results in a table.

    We can now make use of the `run_all_experiments` and `experiment_summary_frame` functions to run and display these experiments.
    """)
    return


@app.cell
def _(experiments_to_run, run_all_experiments):
    results_3 = run_all_experiments(experiments_to_run)

    # illustrate results dataframe.
    results_3["base"].head(2)
    return (results_3,)


@app.cell
def _(results_3):
    results_3["high_demand"].head(2)
    return


@app.cell
def _(experiment_summary_frame, results_3):
    # show results
    # further adaptions might include adding units for figures.
    experiment_summary_frame(results_3).round(2)
    return


@app.cell
def _(experiment_summary_frame, results_3):
    experiment_summary_frame(results_3).round(2).T
    return


if __name__ == "__main__":
    app.run()
