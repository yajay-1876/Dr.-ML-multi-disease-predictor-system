import sys
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.append(PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="Dr.ML Multi Disease Predictor",
    page_icon="🏥",
    layout='centered' 
)

st.title("🧠 Dr. ML - Multi-Disease Predictor")
st.write(
    """
Use the left sidebar to navigate:
- 🩺 Diabetes Risk Predictor
- ❤️ Heart Disease Risk Predictor
"""
)

st.info("Make sure FastAPI backend is running in background")