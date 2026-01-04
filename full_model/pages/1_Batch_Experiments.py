import pandas as pd
import streamlit as st
from app_utility.file_io import read_file_contents
from output_analysis import create_example_csv

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
