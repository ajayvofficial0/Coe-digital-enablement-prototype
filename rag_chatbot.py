import os
import re
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
FALLBACK_ANSWER = "I could not find a reliable answer in the provided documents."

# -----------------------------
# Small helper for repeated junk
# -----------------------------
def clean_generated_answer(text: str) -> str:
    text = text.strip()

    if not text:
        return FALLBACK_ANSWER

    lowered = text.lower()

    if "i could not find a reliable answer" in lowered:
        return FALLBACK_ANSWER

    words = re.findall(r"\b[a-zA-Z]+\b", lowered)
    if len(words) >= 6:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:
            return FALLBACK_ANSWER

    if re.search(r"\b(\w+)(,\s*\1){2,}\b", lowered):
        return FALLBACK_ANSWER

    return text


# -----------------------------
# Load Documents + Build Index
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=60
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"✅ Created {len(split_docs)} chunks")

    print("🔤 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ Embeddings created")

    print("📦 Building FAISS vector store...")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    print("✅ Vector store created")

    return vectorstore


# -----------------------------
# Load Generator Model
# -----------------------------
@st.cache_resource
def load_generator():
    print("🤖 Loading LLM model (FLAN-T5)...")
    model_name = "google/flan-t5-base"
    # If your laptop can handle it, you can try:
    # model_name = "google/flan-t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("✅ LLM loaded")

    return tokenizer, model


vectorstore = load_and_index_documents()
tokenizer, model = load_generator()

# -----------------------------
# Query Input
# -----------------------------
query = st.text_input("Enter your question:")

if query:
    cleaned_query = query.strip()
    print(f"🔎 User query: {cleaned_query}")
    with st.spinner("Retrieving relevant context and generating answer..."):
    
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        print("📡 Retrieving relevant chunks...")
        retrieved_docs = retriever.invoke(cleaned_query)
        print(f"✅ Retrieved {len(retrieved_docs)} chunks")

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""   
Answer the question using only the context below.

Give a clear answer in 2 to 4 complete sentences.
If the question asks about an acronym, explain its full form and how it helps an organization.
If the answer is not in the context, say exactly:
I could not find a reliable answer in the provided documents.

Question: {cleaned_query}

Context:
{context}

Answer:
"""

        print("🧠 Generating answer...")
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            min_new_tokens=20,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        result = clean_generated_answer(result)

    # -----------------------------
    # Confidence Score Heuristic
    # -----------------------------
    retrieved_count = len(retrieved_docs)
    avg_chunk_len = sum(len(doc.page_content) for doc in retrieved_docs) / max(retrieved_count, 1)

    if result == FALLBACK_ANSWER:
        confidence = 0.25
    elif retrieved_count == 3 and avg_chunk_len > 150:
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
    st.divider()

    st.subheader("Confidence Score")
    st.write(f"{confidence * 100:.0f}%")
    st.caption("Confidence is estimated using a simple heuristic based on retrieval quality.")

    st.subheader("Retrieved Context (Top 3 Chunks)")
    for i, doc in enumerate(retrieved_docs, start=1):
        source_name = os.path.basename(doc.metadata.get("source", "Unknown Source"))
        with st.expander(f"Retrieved Source {i} - {source_name}"):
            st.write(doc.page_content)

    print("✅ Output displayed")