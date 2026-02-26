import streamlit as st
import pandas as pd
from openai import OpenAI
from prompt import SYSTEM_PROMPT

# Page config
st.set_page_config(
    page_title="🩺 AI Medical Report Analyzer",
    page_icon="🧠",
    layout="centered"
)

st.title("🩺 AI Medical Report Analyzer")
st.write("Upload a medical report CSV to get AI-powered explanations")

# Load API key from secrets
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OpenAI API key not found. Please add it to Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# Upload CSV
uploaded_file = st.file_uploader("📄 Upload Medical Report (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Medical Data")
    st.dataframe(df)

    if st.button("🧠 Analyze Medical Report"):
        csv_text = df.to_string(index=False)

        with st.spinner("Analyzing report using GenAI..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"""
Here is a patient's medical report in CSV format:

{csv_text}

Explain the results clearly for a patient.
"""
                    }
                ]
            )

        st.success("✅ Analysis Complete")
        st.markdown(response.choices[0].message.content)

st.caption("⚠️ Educational use only. Not a medical diagnosis tool.")
