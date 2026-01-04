import pandas as pd
import streamlit as st
from app_utility.file_io import read_file_contents
from execution import run_all_experiments
from output_analysis import create_example_csv, experiment_summary_frame

from experiment import create_experiments

INFO_1 = "**Execute multiple experiments in a batch**"
INFO_2 = "### Upload a CSV containing input parameters"

st.title("Urgent care call centre")
st.markdown(INFO_1)

with st.expander("Template to use for experiments"):
    st.markdown(read_file_contents("resources/batch_upload_txt.md"))
    template = create_example_csv()
    st.dataframe(template, hide_index=True)

st.markdown(INFO_2)
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df_experiments = pd.read_csv(uploaded_file)
    st.write("Loaded Experiments")
    st.table(df_experiments)

    n_reps = st.slider("Replications", 3, 30, 5, step=1)
    warm_up_period = st.number_input("Warm-up period", 0, 1_000, step=1)
    results_collection_period = st.number_input(
        "Data collection period", 1_000, 10_000, step=1
    )

    if st.button("Execute Experiments"):
        experiments = create_experiments(df_experiments)
        with st.spinner("Running all experiments"):
            results = run_all_experiments(
                experiments, warm_up_period, results_collection_period, n_reps
            )

            st.success("Done")

            df_results = experiment_summary_frame(results)
            st.dataframe(df_results.round(2))
