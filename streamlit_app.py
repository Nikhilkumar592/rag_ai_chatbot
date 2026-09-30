"""Optional Streamlit UI.  Run:  streamlit run streamlit_app.py"""
import streamlit as st

from src import graph as rag
from src.gemini_errors import GEMINI_QUOTA_URL, explain_gemini_error, is_gemini_quota_error

st.set_page_config(page_title="Agentic AI RAG Chatbot", layout="wide")
st.title("Agentic AI eBook - RAG Chatbot")
st.caption("Answers are grounded strictly in the eBook. Out-of-scope questions are refused.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        st.write(turn["final_answer"])

query = st.chat_input("Ask a question about the eBook...")
if query:
    with st.spinner("Retrieving and generating..."):
        try:
            result = rag.ask(query)
        except Exception as exc:
            st.error(explain_gemini_error(exc))
            if is_gemini_quota_error(exc):
                st.link_button("Check Gemini API rate limits", GEMINI_QUOTA_URL)
            else:
                st.info("Check that your Gemini API key, model, and Pinecone settings are correct.")
        else:
            st.session_state.history.append(result)
            st.rerun()

with st.sidebar:
    st.header("Last response")
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.metric("Confidence score", last["confidence_score"])
        st.subheader("Retrieved chunks")
        for i, chunk in enumerate(last["retrieved_context_chunks"], 1):
            with st.expander(f"Chunk {i}"):
                st.write(chunk)
        st.subheader("Raw JSON")
        st.json(last)
    else:
        st.info("Ask something to see retrieved context here.")
