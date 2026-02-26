import streamlit as st
import pandas as pd
from openai import OpenAI
from summarizer import summarize_patient

st.set_page_config(page_title="Healthcare GenAI", page_icon="🏥")

# ✅ Access secrets ONLY here
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🏥 Healthcare GenAI – Medical Report Summarizer")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    patient_id = st.selectbox("Select Patient", df["patient_id"])

    if st.button("Generate Summary"):
        row = df[df["patient_id"] == patient_id].iloc[0]
        summary = summarize_patient(row, client)
        st.success(summary)
