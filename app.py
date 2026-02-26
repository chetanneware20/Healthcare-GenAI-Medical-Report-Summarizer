import streamlit as st
import pandas as pd
from summarizer import summarize_patient

st.set_page_config(page_title="Healthcare GenAI", page_icon="🏥")

if "GEMINI_API_KEY" not in st.secrets:
    st.error("Gemini API key not found in Streamlit Secrets.")
    st.stop()

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

st.title("🏥 Healthcare GenAI – Medical Report Summarizer")

uploaded_file = st.file_uploader("Upload Medical Report CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df, use_container_width=True)

    patient_id = st.selectbox("Select Patient ID", df["patient_id"].unique())

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            row = df[df["patient_id"] == patient_id].iloc[0]
            summary = summarize_patient(row, GEMINI_API_KEY)

        st.subheader("AI / Rule-Based Summary")
        st.success(summary)

st.markdown(
    """
    ⚠️ **Disclaimer**  
    This tool is for educational purposes only.  
    It does not provide medical advice.
    """
)
