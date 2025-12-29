import simulation_constants as sim_const


def trace(msg):
    """
    Turing printing of events on and off.

    Params:
    -------
    msg: str
        string to print to screen.
    """
    if sim_const.TRACE:
        print(msg)
