"""
Simpy application using streamlit package

This application implements a discrete event simulation of an urgent care call centre
"""

import streamlit as st
from app_utility.file_io import read_file_contents

st.title("A DES model of an Urgent care call centre")
st.markdown(read_file_contents("./resources/model_info.md"))
