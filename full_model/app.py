"""
Simpy application using streamlit package

This application implements a discrete event simulation of an urgent care call centre
"""

import streamlit as st
from app_utility.file_io import read_file_contents
from app_utility.results import get_kpi_name_mappings
from execution import multiple_replications
from output_analysis import create_user_controlled_hist

from experiment import Experiment

st.title("A DES model of an Urgent care call centre")
st.markdown(read_file_contents("./resources/model_info.md"))

# no. resources
n_operators = st.slider("Call Operators", 1, 20, 13, step=1)
n_nurses = st.slider("Nurses", 1, 20, 9, step=1)

# demand
mean_iat = st.slider("IAT", 0.1, 1.0, 0.6, step=0.05)

# patient routing
chance_call_back = st.slider("Chance of nurse call back", 0.1, 1.0, 0.4, step=0.05)

# set number of replications
n_reps = st.number_input("No. replications", 100, 1_000, step=1)

# set warm-up period
warm_up_period = st.number_input("Warm-up period", 0, 1_000, step=1)
results_collection_period = st.number_input(
    "Data collection period", 1_000, 10_000, step=1
)

if st.button("Run simulation"):
    user_experiment = Experiment(
        n_operators=n_operators,
        n_nurses=n_nurses,
        mean_iat=mean_iat,
        chance_callback=chance_call_back,
    )

    results = multiple_replications(
        experiment=user_experiment,
        wu_period=warm_up_period,
        rc_period=results_collection_period,
        n_reps=n_reps,
    )

    st.dataframe(results.describe().round(2).T)
    fig = create_user_controlled_hist(results, name_mappings=get_kpi_name_mappings())
    st.plotly_chart(fig)
