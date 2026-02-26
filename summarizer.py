import google.generativeai as genai
import time

def summarize_patient(row, api_key, retries=3):
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    You are a medical report summarization assistant.

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
def fallback_summary(row):
    summary = []

    if row["hba1c"] > 6.5:
        summary.append("Blood sugar levels appear higher than the normal range.")

    if row["fasting_glucose"] > 126:
        summary.append("Fasting glucose value is elevated.")

    if not summary:
        summary.append("No major abnormal values are detected in the report.")

    return " ".join(summary)
    Task:
    - Summarize in simple, patient-friendly language
    - Highlight abnormal values
    - Do NOT give medical advice
    - Do NOT change diagnosis
    """

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            time.sleep(2 ** attempt)

    return "⚠️ Unable to generate summary at the moment. Please try again."
