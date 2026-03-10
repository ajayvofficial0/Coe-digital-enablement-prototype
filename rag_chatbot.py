import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="COE AI Chatbot", layout="wide")
st.title("COE AI Chatbot (RAG-Based)")
st.markdown("Ask questions about Lean, Six Sigma, KPI tracking, and operational excellence.")

DOCS_PATH = "docs"


# -----------------------------
# Load Documents
# -----------------------------
@st.cache_resource
def load_and_index_documents():
    print("📂 Loading documents...")
    documents = []

    for file_name in os.listdir(DOCS_PATH):
        if file_name.endswith(".txt"):
            file_path = os.path.join(DOCS_PATH, file_name)
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
    print(f"✅ Loaded {len(documents)} documents")

    print("✂️ Chunking documents...")
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"✅ Created {len(split_docs)} chunks")

    print("🔤 Creating embeddings...")
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embeddings created")
    # Vector Store
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    print("✅ Vector store created")    
    return vectorstore, split_docs


@st.cache_resource
def load_generator():
    print("🤖 Loading LLM model (FLAN-T5)...")
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("✅ LLM loaded")
    return tokenizer, model

vectorstore, split_docs = load_and_index_documents()
tokenizer, model = load_generator()


# -----------------------------
# Query Input
# -----------------------------
query = st.text_input("Enter your question:")

if query:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Retrieve top 3 documents
    retrieved_docs = retriever.invoke(query)
    print(f"✅ Retrieved {len(retrieved_docs)} chunks")

    # Build context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    You are an enterprise excellence assistant.

    Using the context below, answer the question clearly and completely.

    If the answer includes definitions, frameworks, or steps, list them explicitly.

    Context:
    {context}   

    Question:
    {query}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=150)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Simple confidence score heuristic
    retrieved_count = len(retrieved_docs)
    avg_chunk_len = sum(len(doc.page_content) for doc in retrieved_docs) / max(retrieved_count, 1)

    if retrieved_count == 3 and avg_chunk_len > 150:
        confidence = 0.88
    elif retrieved_count >= 2:
        confidence = 0.74
    else:
        confidence = 0.58
    print(f"✅ Confidence: {confidence * 100:.0f}%")
    # -----------------------------
    # Output
    # -----------------------------
    st.subheader("Answer")
    st.write(result)

    st.subheader("Confidence Score")
    st.progress(confidence)
    st.write(f"{confidence * 100:.0f}%")

    st.subheader("Retrieved Context (Top 3 Chunks)")
    for i, doc in enumerate(retrieved_docs, start=1):
        with st.expander(f"Retrieved Source {i}"):
            st.write(doc.page_content)
    print("✅ Output displayed")    