def get_kpi_name_mappings():
    """
    Returns a dictionary that maps the performance
    measure variable names in the model to
    1. a user "friendly name"
    2. units of measurement.

    Use to improve readability of results.

    Returns:
    --------
    dict{str:{'friendly_name':str, 'units':str}}
    """
    name_mappings = {
        "01_mean_waiting_time": {
            "friendly_name": "Call answer waiting time",
            "units": "minutes",
        },
        "02_operator_util": {
            "friendly_name": "Operator Utilization",
            "units": "%",
        },
        "03_operator_queue_length": {
            "friendly_name": "Operator queue length",
            "units": "Patients",
        },
        "04_mean_nurse_waiting_time": {
            "friendly_name": "Nurse waiting time",
            "units": "minutes",
        },
        "05_nurse_util": {
            "friendly_name": "Nurse Utilization",
            "units": "%",
        },
        "06_nurse_queue_length": {
            "friendly_name": "Nurse queue length",
            "units": "Patients",
        },
    }
    return name_mappings
