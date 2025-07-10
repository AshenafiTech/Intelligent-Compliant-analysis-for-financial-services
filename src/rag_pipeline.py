import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load vector store and metadata
index = faiss.read_index("notebooks/vector_store/complaints_faiss.index")
with open("notebooks/vector_store/metadata.pkl", "rb") as f:
    metadatas = pickle.load(f)

# Load original data for source retrieval
import pandas as pd
df = pd.read_csv("data/processed/filtered_complaints.csv")

# Load embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)

def retrieve_chunks(question, k=5):
    """Embed question and retrieve top-k most similar chunks."""
    q_emb = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    retrieved = []
    for idx in I[0]:
        meta = metadatas[idx]
        chunk_text = df.loc[meta["complaint_id"], "cleaned_narrative"]
        retrieved.append({
            "chunk": chunk_text,
            "meta": meta
        })
    return retrieved

def build_prompt(question, retrieved_chunks):
    """Format the prompt for the LLM."""
    context = "\n\n".join([c["chunk"] for c in retrieved_chunks])
    prompt = (
        "You are a financial analyst assistant for CrediTrust. "
        "Your task is to answer questions about customer complaints. "
        "Use the following retrieved complaint excerpts to formulate your answer. "
        "If the context doesn't contain the answer, state that you don't have enough information.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return prompt

def generate_answer(prompt, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    """Generate answer using an LLM (HuggingFace pipeline example)."""
    from transformers import pipeline
    pipe = pipeline("text-generation", model=model_name, max_new_tokens=256)
    response = pipe(prompt)[0]["generated_text"]
    # Optionally, extract only the answer part
    return response.split("Answer:")[-1].strip()

def rag_qa(question, k=5, llm_model="mistralai/Mistral-7B-Instruct-v0.2"):
    retrieved = retrieve_chunks(question, k)
    prompt = build_prompt(question, retrieved)
    answer = generate_answer(prompt, model_name=llm_model)
    return {
        "question": question,
        "answer": answer,
        "retrieved_sources": retrieved[:2]  # Show 1-2 for report
    }