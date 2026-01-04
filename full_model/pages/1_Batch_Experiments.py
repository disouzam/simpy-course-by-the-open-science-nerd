import streamlit as st
from app_utility.file_io import read_file_contents

INFO_1 = "**Execute multiple experiments in a batch**"
INFO_2 = "### Upload a CSV containing input parameters"

st.title("Urgent care call centre")
st.markdown(INFO_1)

with st.expander("Template to use for experiments"):
    st.markdown(read_file_contents("resources/batch_upload_txt.md"))
