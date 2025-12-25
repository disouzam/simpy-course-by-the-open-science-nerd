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
    # Sampling from statistical distributions in Python

    If you are working in simulation modelling in Python, you will likely need to use `numpy.random` namespace. It provides a variety of statistical distributions which you can use for efficient sampling.

    This notebook will guide you through examples of

    1.  Creating instances of a high quality Pseudo Random Number Generator (PRNG) using PCG64 provided by `numpy`
    2.  Generating samples from the **uniform**, **exponential** and **normal** distributions.
    3.  Spawning multiple non-overlapping streams of random numbers
    4.  Using OOP to encapsulate PRNGs, distributions and parameters for simulation models.

    by The Open Science Nerd at https://github.com/pythonhealthdatascience/intro-open-sim
    """)
    return


if __name__ == "__main__":
    app.run()
