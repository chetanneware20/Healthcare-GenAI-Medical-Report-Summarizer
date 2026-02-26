import streamlit as st
from transformers import pipeline
import PyPDF2

# Page config
st.set_page_config(
    page_title="Healthcare GenAI – Medical Report Summarizer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Healthcare GenAI – Medical Report Summarizer")
st.write("Upload a medical report and get an AI-generated summary.")
st.caption("⚠️ This tool is for educational purposes only and not medical advice.")

# Load summarization model
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

summarizer = load_model()

# File upload
uploaded_file = st.file_uploader(
    "Upload Medical Report (PDF or TXT)",
    type=["pdf", "txt"]
)

def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    else:
        return file.read().decode("utf-8")

if uploaded_file:
    report_text = extract_text(uploaded_file)

    st.subheader("📄 Extracted Report Text")
    with st.expander("View Report"):
        st.write(report_text)

    if st.button("🧠 Generate Summary"):
        with st.spinner("Summarizing report..."):
            summary = summarizer(
                report_text,
                max_length=150,
                min_length=50,
                do_sample=False
            )

        st.subheader("📝 AI Generated Summary")
        st.success(summary[0]["summary_text"])
