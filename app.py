"""
DAY 4 (upgraded): STREAMLIT UI with Multi-Retriever + Re-Ranking
--------------------------------------------------------------------
Farq purani app.py se: "Build Index" ab teeno indexes banata hai (vector,
sentence-window, graph), aur har answer ke sath ye bhi dikhta hai ke
kaunse retrievers ne is jawab mein contribute kiya.
"""

import os
import streamlit as st
from ingestion import build_unified_documents
from indexing import build_vector_index
from sentence_window_retriever import build_sentence_window_index
from graph_retriever import build_knowledge_graph
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

        with st.spinner("Ingesting sources..."):
            docs = build_unified_documents(pdf_paths=pdf_paths, urls=urls, csv_paths=csv_paths)

        if docs:
            with st.spinner("Building vector index..."):
                build_vector_index(docs)
            with st.spinner("Building sentence-window index..."):
                build_sentence_window_index(docs)
            with st.spinner("Building knowledge graph (entities + relations)..."):
                build_knowledge_graph(docs)

            st.session_state.indexed = True
            st.success(f"Indexed {len(docs)} document chunks across 3 retrievers ✅")
        else:
            st.warning("Koi source add nahi ki gayi.")

    st.markdown("---")
    st.markdown("### 🔍 Active Retrievers")
    st.caption("Vector search · Sentence-window · Graph relations")

# ---------------- Main: Chat ----------------
st.title("🧠 Knowledge Base Assistant")
st.caption("Multi-source, multi-retriever RAG — ask questions across your PDFs, websites, and CSVs")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn["role"] == "assistant":
            if turn.get("sources"):
                st.caption("Sources: " + " · ".join(turn["sources"]))
            if turn.get("retrievers_used"):
                st.caption("Retrieved via: " + " + ".join(turn["retrievers_used"]))

query = st.chat_input("Ask a question across your sources...")

if query:
    if not st.session_state.indexed:
        st.warning("Pehle sidebar se koi source add karein aur index build karein.")
    else:
        st.session_state.history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving from vector + sentence-window + graph, then re-ranking..."):
                result = answer_question(query)
                st.write(result["answer"])
                if result["sources"]:
                    st.caption("Sources: " + " · ".join(result["sources"]))
                if result["retrievers_used"]:
                    st.caption("Retrieved via: " + " + ".join(result["retrievers_used"]))

        st.session_state.history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrievers_used": result["retrievers_used"],
        })