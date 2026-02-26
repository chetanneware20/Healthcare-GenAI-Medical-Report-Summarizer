import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Page config
st.set_page_config(
    page_title="Healthcare GenAI – Patient Summary",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Healthcare GenAI – Patient Summary Generator")
st.caption("⚠️ Educational use only. Not medical advice.")

# Load model safely (NO pipeline)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Load CSV
@st.cache_data
def load_data():
    return pd.read_csv("sample_report.csv")

df = load_data()

st.subheader("📊 Patient Records")
st.dataframe(df, use_container_width=True)

# Select patient
patient_id = st.selectbox(
    "Select Patient ID",
    df["patient_id"].unique()
)

patient = df[df["patient_id"] == patient_id].iloc[0]

# Convert structured row → clinical text
clinical_text = f"""
Patient Name: {patient['name']}
Age: {patient['age']}
Gender: {patient['gender']}
Diagnosis: {patient['diagnosis']}
Symptoms: {patient['symptoms']}
HbA1c: {patient['hba1c']}
Blood Pressure: {patient['blood_pressure']}
Fasting Glucose: {patient['fasting_glucose']}
Medication: {patient['medication']}
Follow-up in weeks: {patient['follow_up_weeks']}
Clinical Notes: {patient['notes']}
"""

st.subheader("📄 Clinical Input Text")
with st.expander("View Generated Clinical Text"):
    st.write(clinical_text)

# Summarize using model.generate()
if st.button("🧠 Generate AI Summary"):
    with st.spinner("Generating summary..."):
        inputs = tokenizer(
            clinical_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=60,
                num_beams=4,
                early_stopping=True
            )

        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

    st.subheader("📝 AI Generated Summary")
    st.success(summary)
