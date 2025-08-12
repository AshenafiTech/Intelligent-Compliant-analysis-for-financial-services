import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load cleaned data
df = pd.read_csv('data/filtered_complaints.csv')

# Chunking strategy
chunk_size = 300  # Experimented and found a balance between context and granularity
chunk_overlap = 50

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# Prepare chunks and metadata
chunks = []
metadatas = []
for idx, row in df.iterrows():
    text = str(row['cleaned_narrative'])
    splits = splitter.split_text(text)
    for i, chunk in enumerate(splits):
        chunks.append(chunk)
        metadatas.append({
            "complaint_id": idx,
            "product": row['Product'],
            "chunk_index": i
        })

# Embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)
embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/complaints_faiss.index")
with open("vector_store/metadata.pkl", "wb") as f:
    pickle.dump(metadatas, f)

print(f"Indexed {len(chunks)} chunks. Vector store saved in vector_store/")