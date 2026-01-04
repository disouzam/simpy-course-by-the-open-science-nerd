def read_file_contents(file_name):
    """'
    Read the contents of a file.

    Params:
    ------
    file_name: str
        Path to file.

    Returns:
    -------
    str
    """
    with open(file_name) as f:
        return f.read()
