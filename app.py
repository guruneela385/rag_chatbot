import os
import time
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 RAG Chatbot")

st.write(
    "This chatbot answers questions only from its internal documents. "
    "If a question is outside the knowledge base, it will respond with 'I don't know'."
)

st.markdown("**Example questions you can try:**")
st.markdown("""
- What is RAG?
- How does Retrieval Augmented Generation work?
- What is FAISS?
- What are sentence embeddings?
""")

# -----------------------------
# Load Models & Data
# -----------------------------
@st.cache_resource
def load_models_and_data():
    # Embedding model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load documents
    texts = []
    for file in os.listdir("documents"):
        if file.endswith(".txt"):
            with open(os.path.join("documents", file), "r", encoding="utf-8") as f:
                texts.extend(f.read().split("\n"))

    docs = [t for t in texts if t.strip()]

    # Create embeddings
    embeddings = embed_model.encode(docs)

    # FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Lightweight instruction-tuned LLM
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=150,
        device=-1
    )

    return embed_model, docs, index, llm

embed_model, docs, index, llm = load_models_and_data()

# -----------------------------
# Retriever
# -----------------------------
def retrieve(query, top_k=3):
    query_embedding = embed_model.encode([query])
    _, indices = index.search(query_embedding, top_k)
    return [docs[i] for i in indices[0]]

# -----------------------------
# RAG Generator
# -----------------------------
def generate_answer(query):
    # Normalize common acronym
    query = query.lower().replace("rag", "Retrieval Augmented Generation")

    context_docs = retrieve(query)
    context = "\n".join(context_docs)

    prompt = f"""
You are an AI assistant.

RAG stands for Retrieval Augmented Generation.

Use ONLY the context below to answer the question.
If the answer is not present in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    start_time = time.time()
    response = llm(prompt)[0]["generated_text"]
    latency = time.time() - start_time

    return response, context, latency

# -----------------------------
# UI Interaction
# -----------------------------
user_query = st.text_input("Ask a question:")

if user_query:
    with st.spinner("Thinking..."):
        answer, context, latency = generate_answer(user_query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved Context")
    st.write(context)

    st.caption(f"Response generated in {latency:.2f} seconds")
