import runpy
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Scoring & Backtest", layout="wide")

# ---- Header: Titre + logo à droite ----
LOGO_PATH = Path(__file__).parent / "logo.png"   # mets ton logo ici, ex: "static/logo.png"

left, right = st.columns([1, 0.12])  # ajuste le ratio pour coller le logo à droite
with left:
    st.markdown("<h1 style='margin:0'>Scoring & Backtest</h1>", unsafe_allow_html=True)
with right:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=200)
    else:
        st.warning(f"Logo introuvable: {LOGO_PATH}")

def run_streamlit_script(path: str):
    import streamlit as st
    original = st.set_page_config
    try:
        st.set_page_config = lambda *a, **k: None
        runpy.run_path(path, run_name="__main__")
    finally:
        st.set_page_config = original

# Tabs
tab1, tab2 = st.tabs(["Scoring", "Backtest"])
with tab1:
    run_streamlit_script("scoring.py")
with tab2:
    run_streamlit_script("backtest.py")
