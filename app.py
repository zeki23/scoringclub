# app.py
import runpy
import streamlit as st

st.set_page_config(page_title="Scoring & Backtest", layout="wide")
st.title("Scoring & Backtest")

def run_streamlit_script(path: str):
    """
    Exécute un script Streamlit dans le même process
    en neutralisant set_page_config pour éviter les conflits.
    """
    import streamlit as st
    original = st.set_page_config
    try:
        st.set_page_config = lambda *a, **k: None  # désactive temporairement
        runpy.run_path(path, run_name="__main__")
    finally:
        st.set_page_config = original

# Deux onglets : Scoring / Backtest
tab1, tab2 = st.tabs(["Scoring", "Backtest"])

with tab1:
    run_streamlit_script("scoring.py")

with tab2:
    run_streamlit_script("backtest.py")
