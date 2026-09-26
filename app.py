"""
DAY 4: STREAMLIT UI
----------------------
Ye woh Step 1 + Step 10 hai diagram mein (User Query -> Final Response).
Run karne ke liye:  streamlit run app.py
"""

import os
import streamlit as st
from ingestion import build_unified_documents
from indexing import build_vector_index
from qa_engine import answer_question

st.set_page_config(page_title="Knowledge Base Assistant", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = os.path.exists("data/vectorstore/index.faiss")

# ---------------- Sidebar: Sources + Ingestion ----------------
with st.sidebar:
    st.markdown("### 📚 Sources")

    uploaded_pdfs = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    website_url = st.text_input("Website URL")
    uploaded_csv = st.file_uploader("Upload CSV", type="csv")

    if st.button("Build / Rebuild Index", use_container_width=True):
        pdf_paths = []
        for f in uploaded_pdfs or []:
            path = os.path.join("data/pdfs", f.name)
            with open(path, "wb") as out:
                out.write(f.read())
            pdf_paths.append(path)

        csv_paths = []
        if uploaded_csv is not None:
            path = os.path.join("data", uploaded_csv.name)
            with open(path, "wb") as out:
                out.write(uploaded_csv.read())
            csv_paths.append(path)

        urls = [website_url] if website_url else []

        with st.spinner("Ingesting sources and building index..."):
            docs = build_unified_documents(pdf_paths=pdf_paths, urls=urls, csv_paths=csv_paths)
            if docs:
                build_vector_index(docs)
                st.session_state.indexed = True
                st.success(f"Indexed {len(docs)} document chunks ✅")
            else:
                st.warning("Koi source add nahi ki gayi.")

# ---------------- Main: Chat ----------------
st.title("🧠 Knowledge Base Assistant")
st.caption("Multi-source RAG — ask questions across your PDFs, websites, and CSVs")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            st.caption("Sources: " + " · ".join(turn["sources"]))

query = st.chat_input("Ask a question across your sources...")

if query:
    if not st.session_state.indexed:
        st.warning("Pehle sidebar se koi source add karein aur index build karein.")
    else:
        st.session_state.history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer_question(query)
                st.write(result["answer"])
                if result["sources"]:
                    st.caption("Sources: " + " · ".join(result["sources"]))

        st.session_state.history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
