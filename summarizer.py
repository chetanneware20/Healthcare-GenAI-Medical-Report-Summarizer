import google.generativeai as genai

def fallback_summary(row):
    summary = []

    if row["hba1c"] > 6.5:
        summary.append("Blood sugar levels are higher than the normal range.")

    if row["fasting_glucose"] > 126:
        summary.append("Fasting glucose value is elevated.")

    if row["blood_pressure"] not in ["120/80", "118/76"]:
        summary.append("Blood pressure is outside the typical healthy range.")

    if not summary:
        summary.append("No major abnormal values are observed in this report.")

    return " ".join(summary)


def summarize_patient(row, api_key):
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Rewrite the following medical data in simple, neutral language.

    Rules:
    - Do NOT give medical advice
    - Do NOT suggest treatment
    - Do NOT add new information
    - Only summarize what is already present

    Data:
    Age: {row['age']}
    Gender: {row['gender']}
    Diagnosis: {row['diagnosis']}
    Symptoms: {row['symptoms']}
    HbA1c: {row['hba1c']}
    Blood Pressure: {row['blood_pressure']}
    Fasting Glucose: {row['fasting_glucose']}
    Medication: {row['medication']}
    Notes: {row['notes']}
    """

    try:
        response = model.generate_content(prompt)

        if response and hasattr(response, "text") and response.text.strip():
            return response.text.strip()

        # Gemini blocked or returned empty
        return fallback_summary(row)

    except Exception:
        return fallback_summary(row)
