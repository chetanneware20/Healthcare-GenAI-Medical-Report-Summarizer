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
