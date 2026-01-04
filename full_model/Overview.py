import streamlit as st

#  update to wide page settings to help display results side by side
st.set_page_config(
    page_title="Urgent Care SimPy App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# File paths
PROCESS_IMG = "resources/call_centre_process.png"

# Text to display
INFO_1 = "**A simple simulation model of a urgent care call centre.**"
INFO_2 = """Patient callers arrive at random and wait to be triaged by
a **call operator**.  Once triage is complete a proportion wait to a
call back from a **nurse** and then undergo nurse consultation.
"""

st.title("A Discrete-Event Simulation of an Urgent Care Call Centre")
st.markdown(INFO_1)
st.markdown(INFO_2)

with st.expander("The patient process.", expanded=True):
    st.image(PROCESS_IMG, width="stretch")
