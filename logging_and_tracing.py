"""
Store functionality to log and write tracing information in different outputs
"""


def trace(msg, enabled=False):
    """
    Turning printing of events on and off.

    Params:
    -------
    msg: str
        string to print to screen.
    """
    if enabled:
        print(msg)
