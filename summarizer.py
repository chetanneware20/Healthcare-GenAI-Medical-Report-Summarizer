import streamlit as st
import pandas as pd
from openai import OpenAI

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Healthcare GenAI Medical Report Summarizer",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Healthcare GenAI – Medical Report Summarizer")
st.write("Upload a medical CSV file and generate patient-friendly summaries using GenAI")

# ---------------- OPENAI CLIENT ----------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------- FUNCTIONS ----------------
def summarize_patient(row):
    prompt = f"""
    Patient Details:
    Age: {row['age']}
    Gender: {row['gender']}
    Diagnosis: {row['diagnosis']}
    Symptoms: {row['symptoms']}
    HbA1c: {row['hba1c']}
    Blood Pressure: {row['blood_pressure']}
    Fasting Glucose: {row['fasting_glucose']}
    Medication: {row['medication']}
    Notes: {row['notes']}

    Task:
    - Summarize this medical report in simple, patient-friendly language
    - Highlight abnormal values
    - Do NOT give medical advice
    - Do NOT change or add diagnosis
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a safe medical report summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# ---------------- UI ----------------
uploaded_file = st.file_uploader("📄 Upload Medical Report CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Medical Data")
    st.dataframe(df, use_container_width=True)

    patient_id = st.selectbox(
        "Select Patient ID",
        df["patient_id"].unique()
    )

    if st.button("🧠 Generate AI Summary"):
        with st.spinner("Generating summary..."):
            row = df[df["patient_id"] == patient_id].iloc[0]
            summary = summarize_patient(row)

        st.subheader("📝 AI Generated Summary")
        st.success(summary)

# ---------------- DISCLAIMER ----------------
st.markdown(
    """
    ⚠️ **Disclaimer**  
    This application is for educational purposes only.  
    It does not provide medical advice or diagnosis.
    """
)
