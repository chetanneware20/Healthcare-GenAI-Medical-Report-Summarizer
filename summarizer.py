def summarize_patient(row, client):
    prompt = f"""
    Age: {row['age']}
    Gender: {row['gender']}
    Diagnosis: {row['diagnosis']}
    Symptoms: {row['symptoms']}
    HbA1c: {row['hba1c']}
    Blood Pressure: {row['blood_pressure']}
    Fasting Glucose: {row['fasting_glucose']}
    Medication: {row['medication']}
    Notes: {row['notes']}

    Summarize in simple language.
    Do NOT give medical advice.
    Highlight abnormal values.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a safe medical summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
