import streamlit as st
import pandas as pd
from summarizer import summarize_patient

st.set_page_config(
    page_title="Healthcare GenAI – Gemini",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Healthcare GenAI – Medical Report Summarizer (Gemini)")
st.write("Powered by Google Gemini ✨")

# Load Gemini API Key
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

uploaded_file = st.file_uploader("📄 Upload Medical Report CSV", type=["csv"])

if "summary" not in st.session_state:
    st.session_state.summary = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.dataframe(df, use_container_width=True)

    patient_id = st.selectbox("Select Patient ID", df["patient_id"].unique())

    if st.button("🧠 Generate AI Summary"):
        st.session_state.summary = None
        with st.spinner("Generating summary using Gemini..."):
            row = df[df["patient_id"] == patient_id].iloc[0]
            st.session_state.summary = summarize_patient(row, GEMINI_API_KEY)

    if st.session_state.summary:
        st.subheader("📝 AI Generated Summary")
        st.success(st.session_state.summary)

st.markdown(
    """
    ⚠️ **Disclaimer**  
    This application is for educational purposes only.  
    It does not provide medical advice or diagnosis.
    """
)
