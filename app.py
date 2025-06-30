import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

st.set_page_config(page_title="📄 PDF Summarizer", page_icon="📘")
st.title("📄 Document Summarization Tool")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "".join([page.get_text() for page in pdf])

    st.subheader("📃 Extracted Text")
    st.text_area("Document Content", text, height=300)

    if st.button("🧠 Summarize"):
        with st.spinner("Summarizing..."):
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            summaries = []
            for chunk in chunks:
                summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
                summaries.append(summary)
            final_summary = "\n\n".join(summaries)
        st.subheader("📝 Summary")
        st.write(final_summary)
