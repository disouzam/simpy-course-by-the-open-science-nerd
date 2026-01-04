import streamlit as st
from app_utility.file_io import read_file_contents

LICENSE_FILE = "resources/license.md"

st.markdown(read_file_contents(LICENSE_FILE))
