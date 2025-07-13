import streamlit as st
from utils import document_loader, summarizer, qa, challenge

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("📄 Smart Assistant for Research Summarization")

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    file_name = uploaded_file.name

    # Load document
    if file_name.endswith(".pdf"):
        doc_text = document_loader.load_pdf(uploaded_file)
    else:
        doc_text = document_loader.load_txt(uploaded_file)

    st.subheader("🔹 Document Summary")
    summary = summarizer.summarize_doc(doc_text)
    st.write(summary)

    # Build QA chain
    qa_chain = qa.build_qa_chain(doc_text)

    # Modes
    mode = st.radio("Select Mode", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        user_q = st.text_input("Ask a question based on the document")
        if user_q:
            result = qa_chain(user_q)
            st.write("**Answer:**", result['result'])
            if result['source_documents']:
                st.write("**Reference:** Found in document chunk.")

    else:  # Challenge Me
        st.write("Generating 3 logic-based questions...")
        questions = challenge.generate_questions(doc_text)
        qs = questions.content.split("\n")
        for i, q in enumerate(qs):
            st.write(f"**Q{i+1}. {q}**")
            ua = st.text_input(f"Your answer for Q{i+1}")
            if ua:
                eval_result = challenge.evaluate_answer(q, ua, doc_text)
                st.write(eval_result.content)
