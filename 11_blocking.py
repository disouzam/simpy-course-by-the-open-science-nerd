import marimo

__generated_with = "0.19.4"
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
    # Sequential Resource Holding and Blocking in SimPy

    In this notebook we will learn how to code simpy logic to mimic a process holding a resource while queuing for another resource.  This simulates a blocking scenario in a process.

    We will work with a hypothetical stroke pathway where patients undergo acute treatment followed by transfer at a different hospital to undergo rehabilitation.  A patient must remain in an acute stroke bed until a rehabilitation bed is free.

    > In this example we will not concern ourselves with a warm-up period or initial conditions.

    ![model image](public/bed_blocking_image.png "Bed blocking example")
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Imports
    """)
    return


@app.cell
def _():
    import itertools

    import colored as cd
    import numpy as np
    import simpy

    return cd, itertools, np, simpy


@app.cell
def _():
    # to reduce code these classes can be found in distribution.py
    from distributions import Exponential, Lognormal

    return Exponential, Lognormal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Constants
    """)
    return


@app.cell
def _():
    # default mean inter-arrival times(exp)
    # time is in days
    IAT_STROKES = 1.5

    # resources
    N_ACUTE_BEDS = 9
    N_REHAB_BEDS = 15

    # Acute LoS (Lognormal)
    ACUTE_LOS_MEAN = 7.0
    ACUTE_LOC_STD = 1.0

    # Rehab LoS (Lognormal)
    REHAB_LOS_MEAN = 30.0
    REHAB_LOC_STD = 5.0

    # sampling settings
    N_STREAMS = 3
    DEFAULT_RND_SET = 0

    # run variables (units = days)
    RUN_LENGTH = 100
    return (
        ACUTE_LOC_STD,
        ACUTE_LOS_MEAN,
        DEFAULT_RND_SET,
        IAT_STROKES,
        N_ACUTE_BEDS,
        N_REHAB_BEDS,
        N_STREAMS,
        REHAB_LOC_STD,
        REHAB_LOS_MEAN,
        RUN_LENGTH,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Helper classes and functions
    """)
    return


@app.function
def trace(msg, trace_enabled: False):
    """
    Turing printing of events on and off.

    Params:
    -------
    msg: str
        string to print to screen.
    """
    if trace_enabled:
        print(msg)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Experiment class
    """)
    return


@app.cell
def _(
    ACUTE_LOC_STD,
    ACUTE_LOS_MEAN,
    DEFAULT_RND_SET,
    Exponential,
    IAT_STROKES,
    Lognormal,
    N_ACUTE_BEDS,
    N_REHAB_BEDS,
    N_STREAMS,
    REHAB_LOC_STD,
    REHAB_LOS_MEAN,
    np,
):
    class Experiment:
        """
        Encapsulates the concept of an experiment 🧪 for the stroke pathway
        bed blocking simulator. Manages parameters, PRNG streams and results.
        """

        def __init__(
            self,
            random_number_set=DEFAULT_RND_SET,
            n_streams=N_STREAMS,
            iat_strokes=IAT_STROKES,
            acute_los_mean=ACUTE_LOS_MEAN,
            acute_los_std=ACUTE_LOC_STD,
            rehab_los_mean=REHAB_LOS_MEAN,
            rehab_los_std=REHAB_LOC_STD,
            n_acute_beds=N_ACUTE_BEDS,
            n_rehab_beds=N_REHAB_BEDS,
        ):
            """
            The init method sets up our defaults.
            """
            # sampling
            self.random_number_set = random_number_set
            self.n_streams = n_streams

            # store parameters for the run of the model
            self.iat_strokes = iat_strokes
            self.acute_los_mean = acute_los_mean
            self.acute_los_std = acute_los_std
            self.rehab_los_mean = rehab_los_mean
            self.rehab_los_std = rehab_los_std

            #  place holder for resources
            self.acute_ward = None
            self.rehab_unit = None
            self.n_acute_beds = n_acute_beds
            self.n_rehab_beds = n_rehab_beds

            # initialise results to zero
            self.init_results_variables()

            # initialise sampling objects
            self.init_sampling()

        def set_random_no_set(self, random_number_set):
            """
            Controls the random sampling
            Parameters:
            ----------
            random_number_set: int
                Used to control the set of pseudo random numbers used by
                the distributions in the simulation.
            """
            self.random_number_set = random_number_set
            self.init_sampling()

        def init_sampling(self):
            """
            Create the distributions used by the model and initialise
            the random seeds of each.
            """
            # produce n non-overlapping streams
            seed_sequence = np.random.SeedSequence(self.random_number_set)
            self.seeds = seed_sequence.spawn(self.n_streams)

            # create distributions

            # inter-arrival time distributions
            self.arrival_strokes = Exponential(
                self.iat_strokes, random_seed=self.seeds[0]
            )

            self.acute_los = Lognormal(
                self.acute_los_mean, self.acute_los_std, random_seed=self.seeds[1]
            )

            self.rehab_los = Lognormal(
                self.rehab_los_mean, self.rehab_los_std, random_seed=self.seeds[2]
            )

        def init_results_variables(self):
            """
            Initialise all of the experiment variables used in results
            collection.  This method is called at the start of each run
            of the model
            """
            # variable used to store results of experiment
            self.results = {}
            self.results["n_arrivals"] = 0
            self.results["waiting_acute"] = []
            self.results["bed_blocking_times"] = []

    return (Experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5. Pathway process logic

    The key things to recognise are

    * We request the bed from the acute stroke unit as usual using a `with` context manager
    * We request the rehab bed within the acute bed `with` context manager.  This means we do not release the acute bed while the patient waits for rehab.
    * As we do not use a `with` context manager for rehab there is no teardown and we need to manually release the rehab bed.
    """)
    return


@app.cell
def _(cd):
    def patient_pathway(patient_id, trace_enabled, env, args):
        """Process a patient through the acute ward and rehab unit.
        Simpy generator function.

        Parameters:
        -----------
        patient_id: int
            A unique id representing the patient in the process

        env: simpy.Environment
            The simulation environment

        args: Experiment
            Container class for the simulation parameters/results.
        """
        arrival_time = env.now

        with args.acute_ward.request() as acute_bed_request:
            yield acute_bed_request

            acute_admit_time = env.now
            wait_for_acute = acute_admit_time - arrival_time
            args.results["waiting_acute"].append(wait_for_acute)

            if wait_for_acute < 0.01:
                trace(
                    f"{env.now:.2f}: {cd.Fore.white}{cd.Back.green}Patient {patient_id} admitted to acute ward."
                    + f"(IMMEDIATE ADMISSION){cd.Style.reset}",
                    trace_enabled,
                )
            else:
                trace(
                    f"{env.now:.2f}: {cd.Fore.white}{cd.Back.red}Patient {patient_id} admitted to acute ward."
                    + f"(waited {wait_for_acute:.2f} days){cd.Style.reset}",
                    trace_enabled,
                )

            # Simulate acute care treatment
            acute_care_duration = args.acute_los.sample()
            yield env.timeout(acute_care_duration)

            # Patient is now medically ready for rehabilitation
            medically_ready_time = env.now
            trace(
                f"{env.now:.2f}: {cd.Fore.black}{cd.Back.yellow}Patient {patient_id} medically ready for rehab{cd.Style.reset}",
                trace_enabled,
            )

            # Request a rehab bed but don't release the acute bed immediately
            # Note we are still within the "with" context manager for the acute bed
            # This is where bed blocking occurs. We wait here until the rehab bed
            # is available. Make sure the indentation is correct or you will release
            rehab_bed = args.rehab_unit.request()
            yield rehab_bed

            # Now we have a rehab bed, we can transfer the patient
            transfer_time = env.now
            bed_blocking_duration = transfer_time - medically_ready_time
            args.results["bed_blocking_times"].append(bed_blocking_duration)

            if bed_blocking_duration < 0.01:
                trace(
                    f"{env.now:.2f}: {cd.Fore.black}{cd.Back.dark_sea_green_3b}Patient {patient_id} transferred to rehab. "
                    + f"(IMMEDIATE TRANSFERENCE TO REHAB. No blocking of acute bed){cd.Style.reset}",
                    trace_enabled,
                )
            else:
                trace(
                    f"{env.now:.2f}: {cd.Fore.white}{cd.Back.rosy_brown}Patient {patient_id} transferred to rehab. "
                    + f"(blocked acute bed for {bed_blocking_duration:.2f} days){cd.Style.reset}",
                    trace_enabled,
                )

        # Acute bed is now released
        # Note the indentation!  We are now outside of the with context manager.
        # This automatically releases the simpy resource.

        # Simulate rehabilitation stay
        rehab_duration = args.rehab_los.sample()
        yield env.timeout(rehab_duration)

        # Patient completes rehabilitation and is discharged
        discharge_time = env.now

        # Note: we need to explicitly call release on the rehab resource.
        args.rehab_unit.release(rehab_bed)

        trace(
            f"{discharge_time:.2f}: Patient {patient_id} discharged from Rehab.",
            trace_enabled,
        )

    return (patient_pathway,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6. Arrivals generator

    This is a standard arrivals generator. We create stroke arrivals according to their distribution.
    """)
    return


@app.cell
def _(itertools, patient_pathway):
    def stroke_arrivals_generator(env, trace_enabled, args):
        """
        Arrival process for strokes.

        Parameters:
        ------
        env: simpy.Environment
            The simpy environment for the simulation

        args: Experiment
            The settings and input parameters for the simulation.
        """
        # use itertools as it provides an infinite loop
        # with a counter variable that we can use for unique Ids
        for patient_id in itertools.count(start=1):
            # the sample distribution is defined by the experiment.
            inter_arrival_time = args.arrival_strokes.sample()
            yield env.timeout(inter_arrival_time)

            args.results["n_arrivals"] = patient_id

            trace(
                f"{env.now:.2f}: Patient {patient_id} with stroke arrived.",
                trace_enabled,
            )

            # patient enters pathway
            env.process(patient_pathway(patient_id, trace_enabled, env, args))

    return (stroke_arrivals_generator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 7. Single run function
    """)
    return


@app.cell
def _(RUN_LENGTH, np, simpy, stroke_arrivals_generator):
    def single_run(experiment, rep=0, run_length=RUN_LENGTH, trace_enabled=False):
        """
        Perform a single run of the model and return the results

        Parameters:
        -----------

        experiment: Experiment
            The experiment/paramaters to use with model

        rep: int
            The replication number.

        rc_period: float, optional (default=RUN_LENGTH)
            The run length of the model
        """

        # reset all results variables to zero and empty
        experiment.init_results_variables()

        # set random number set to the replication no.
        # this controls sampling for the run.
        experiment.set_random_no_set(rep)

        # environment is (re)created inside single run
        env = simpy.Environment()

        # simpy resources
        experiment.acute_ward = simpy.Resource(env, experiment.n_acute_beds)
        experiment.rehab_unit = simpy.Resource(env, experiment.n_rehab_beds)

        # we pass all arrival generators to simpy
        env.process(stroke_arrivals_generator(env, trace_enabled, experiment))

        # run model
        env.run(until=run_length)

        # quick stats
        results = {}
        results["mean_acute_wait"] = np.array(
            experiment.results["waiting_acute"]
        ).mean()
        results["mean_bed_blocking"] = np.array(
            experiment.results["bed_blocking_times"]
        ).mean()

        # return single run results
        return results

    return (single_run,)


@app.cell
def _(Experiment, single_run):
    experiment = Experiment()
    results = single_run(experiment, trace_enabled=True)
    print(f"\n{results}")
    return


if __name__ == "__main__":
    app.run()
