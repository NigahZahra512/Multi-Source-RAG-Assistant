"""
FASTAPI BACKEND
------------------
3 kaam karta hai:
  1. /api/build-index + /api/sources -> sources ko PERSISTENTLY track karta hai,
     taake unhe baad mein remove kiya ja sake aur index dubara ban sake.
  2. /api/ask -> sawal ka jawab deta hai, aur conversation history save karta hai.
  3. /api/conversations -> chat history (Claude jaisi) - save/list/load/delete.
"""

import os
import json
import shutil
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ingestion import build_unified_documents, save_full_csv_dump
from indexing import build_vector_index
from sentence_window_retriever import build_sentence_window_index
from graph_retriever import build_knowledge_graph
from qa_engine import answer_question

os.makedirs("data/pdfs", exist_ok=True)

SOURCES_FILE = "data/indexed_sources.json"
CONVERSATIONS_FILE = "data/conversations.json"

app = FastAPI(title="Knowledge Base Assistant API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


is_indexed = bool(load_json(SOURCES_FILE, []))


def rebuild_all_indexes():
    global is_indexed
    sources = load_json(SOURCES_FILE, [])

    pdf_paths = [s["path"] for s in sources if s["type"] == "pdf"]
    urls = [s["path"] for s in sources if s["type"] == "url"]
    csv_paths = [s["path"] for s in sources if s["type"] == "csv"]

    if not pdf_paths and not urls and not csv_paths:
        is_indexed = False
        return 0

    docs = build_unified_documents(pdf_paths=pdf_paths, urls=urls, csv_paths=csv_paths)
    save_full_csv_dump(docs)
    build_vector_index(docs)
    build_sentence_window_index(docs)
    build_knowledge_graph(docs)
    is_indexed = True
    return len(docs)


@app.post("/api/build-index")
async def build_index(
    pdfs: list[UploadFile] = File(default=[]),
    url: str = Form(default=""),
    csv: UploadFile = File(default=None),
):
    sources = load_json(SOURCES_FILE, [])

    for f in pdfs:
        path = os.path.join("data/pdfs", f.filename)
        with open(path, "wb") as out:
            shutil.copyfileobj(f.file, out)
        sources.append({"type": "pdf", "name": f.filename, "path": path})

    if url:
        sources.append({"type": "url", "name": url, "path": url})

    if csv is not None and csv.filename:
        path = os.path.join("data", csv.filename)
        with open(path, "wb") as out:
            shutil.copyfileobj(csv.file, out)
        sources.append({"type": "csv", "name": csv.filename, "path": path})

    save_json(SOURCES_FILE, sources)

    chunk_count = rebuild_all_indexes()
    if chunk_count == 0:
        return {"success": False, "message": "Koi source nahi mili."}

    return {"success": True, "chunks_indexed": chunk_count}


@app.get("/api/sources")
async def get_sources():
    return {"sources": load_json(SOURCES_FILE, [])}


@app.delete("/api/sources/{index}")
async def delete_source(index: int):
    sources = load_json(SOURCES_FILE, [])
    if 0 <= index < len(sources):
        removed = sources.pop(index)
        if removed["type"] in ("pdf", "csv") and os.path.exists(removed["path"]):
            os.remove(removed["path"])
        save_json(SOURCES_FILE, sources)
        chunk_count = rebuild_all_indexes()
        return {"success": True, "chunks_indexed": chunk_count, "sources": sources}
    return {"success": False, "message": "Source not found."}


@app.post("/api/conversations")
async def create_conversation():
    conversations = load_json(CONVERSATIONS_FILE, {})
    conv_id = str(uuid.uuid4())[:8]
    conversations[conv_id] = {
        "title": "New chat",
        "created_at": datetime.utcnow().isoformat(),
        "messages": [],
    }
    save_json(CONVERSATIONS_FILE, conversations)
    return {"id": conv_id}


@app.get("/api/conversations")
async def list_conversations():
    conversations = load_json(CONVERSATIONS_FILE, {})
    items = [{"id": cid, "title": c["title"], "created_at": c["created_at"]} for cid, c in conversations.items()]
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return {"conversations": items}


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conversations = load_json(CONVERSATIONS_FILE, {})
    return conversations.get(conv_id, {"title": "New chat", "messages": []})


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    conversations = load_json(CONVERSATIONS_FILE, {})
    conversations.pop(conv_id, None)
    save_json(CONVERSATIONS_FILE, conversations)
    return {"success": True}


@app.post("/api/ask")
async def ask(payload: dict):
    query = payload.get("question", "").strip()
    conv_id = payload.get("conversation_id")

    if not is_indexed:
        return {"answer": "Pehle sidebar se koi source add kar ke index build karein.",
                "sources": [], "retrievers_used": [], "retrieved_chunks": []}
    if not query:
        return {"answer": "Koi sawal likhein.", "sources": [], "retrievers_used": [], "retrieved_chunks": []}

    conversations = load_json(CONVERSATIONS_FILE, {})
    conv = conversations.get(conv_id, {
        "title": "New chat",
        "created_at": datetime.utcnow().isoformat(),
        "messages": [],
    })

    history = [{"role": m["role"], "content": m["content"]} for m in conv["messages"]]
    result = answer_question(query, history=history)

    conv["messages"].append({"role": "user", "content": query})
    conv["messages"].append({
        "role": "assistant", "content": result["answer"],
        "sources": result["sources"], "retrievers_used": result["retrievers_used"],
        "retrieved_chunks": result["retrieved_chunks"],
    })
    if conv["title"] == "New chat":
        conv["title"] = query[:40] + ("..." if len(query) > 40 else "")

    if conv_id:
        conversations[conv_id] = conv
        save_json(CONVERSATIONS_FILE, conversations)

    return result


app.mount("/", StaticFiles(directory="static", html=True), name="static")