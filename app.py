import streamlit as st
from summarizer import summarize_report

st.set_page_config(page_title="Medical Report Summarizer 🏥")

st.title("🏥 GenAI Medical Report Summarizer")

uploaded_file = st.file_uploader("Upload Medical Report", type=["txt", "pdf"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    if st.button("Generate Summary"):
        summary = summarize_report(text)
        st.success("Summary Generated")
        st.write(summary)
