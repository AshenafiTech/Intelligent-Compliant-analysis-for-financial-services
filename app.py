import streamlit as st
from src.rag_pipeline import rag_qa

st.set_page_config(page_title="CrediTrust Complaint RAG", page_icon="💬")

st.title("CrediTrust Complaint Analyst 💬")
st.write("Ask any question about customer complaints. The AI will answer using real complaint excerpts.")

if "history" not in st.session_state:
    st.session_state["history"] = []

def clear_history():
    st.session_state["history"] = []

with st.form(key="qa_form"):
    user_question = st.text_input("Your question:", "")
    submit = st.form_submit_button("Ask")
    clear = st.form_submit_button("Clear", on_click=clear_history)

if submit and user_question.strip():
    result = rag_qa(user_question)
    st.session_state["history"].append({
        "question": user_question,
        "answer": result["answer"],
        "sources": result["retrieved_sources"]
    })

for entry in reversed(st.session_state["history"]):
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**AI:** {entry['answer']}")
    with st.expander("Show sources"):
        for i, src in enumerate(entry["sources"]):
            st.markdown(f"**Source {i+1}:** {src['chunk'][:500]}...")  # Show up to 500 chars

st.info("Powered by your RAG pipeline. Sources are shown for transparency.")
