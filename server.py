"""
FASTAPI BACKEND
------------------
Ye Streamlit ki jagah leta hai. Ye 2 main kaam karta hai:
  1. /api/build-index -> sources le kar teeno indexes banata hai
  2. /api/ask         -> sawal + conversation history le kar answer_question() call karta hai
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ingestion import build_unified_documents, save_full_csv_dump
from indexing import build_vector_index
from sentence_window_retriever import build_sentence_window_index
from graph_retriever import build_knowledge_graph
from qa_engine import answer_question

os.makedirs("data/pdfs", exist_ok=True)

app = FastAPI(title="Knowledge Base Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

is_indexed = False


@app.post("/api/build-index")
async def build_index(
    pdfs: list[UploadFile] = File(default=[]),
    url: str = Form(default=""),
    csv: UploadFile = File(default=None),
):
    global is_indexed

    pdf_paths = []
    for f in pdfs:
        path = os.path.join("data/pdfs", f.filename)
        with open(path, "wb") as out:
            shutil.copyfileobj(f.file, out)
        pdf_paths.append(path)

    csv_paths = []
    if csv is not None and csv.filename:
        path = os.path.join("data", csv.filename)
        with open(path, "wb") as out:
            shutil.copyfileobj(csv.file, out)
        csv_paths.append(path)

    urls = [url] if url else []

    docs = build_unified_documents(pdf_paths=pdf_paths, urls=urls, csv_paths=csv_paths)
    if not docs:
        return {"success": False, "message": "Koi source nahi mili."}

    save_full_csv_dump(docs)

    build_vector_index(docs)
    build_sentence_window_index(docs)
    build_knowledge_graph(docs)
    is_indexed = True

    return {"success": True, "chunks_indexed": len(docs)}


@app.post("/api/ask")
async def ask(payload: dict):
    query = payload.get("question", "").strip()
    history = payload.get("history", [])

    if not is_indexed:
        return {
            "answer": "Pehle sidebar se koi source add kar ke index build karein.",
            "sources": [], "retrievers_used": [], "retrieved_chunks": [],
        }
    if not query:
        return {"answer": "Koi sawal likhein.", "sources": [], "retrievers_used": [], "retrieved_chunks": []}

    result = answer_question(query, history=history)
    return result


app.mount("/", StaticFiles(directory="static", html=True), name="static")