# Advanced Multi-Source RAG — Working v1

Ye Day 1-2-4 ka **working MVP** hai — PDF/website/CSV ingest karo, index bano,
sawal pucho, citations ke sath jawab milega. Day 3 (multi-retriever + graph +
re-ranking) hum agle step mein isi project ke upar add karenge.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`.env.example` ko `.env` naam de kar apni Groq API key paste karein.

## Run

```bash
streamlit run app.py
```

Browser mein `http://localhost:8501` khul jayega.

## File Guide

| File | Roadmap Step | Kaam |
|---|---|---|
| `ingestion.py` | Day 1 | PDF / website / CSV se text nikal kar unify karta hai |
| `indexing.py` | Day 2 | Chunking + embeddings + FAISS vector index |
| `qa_engine.py` | Day 4 (core) | Retrieval + LLM se answer + citations |
| `app.py` | Day 4 (UI) | Streamlit interface |

## Next (Day 3 — abhi baaki hai)

- Sentence-window retriever add karna
- Graph-based retriever (entities/relations) add karna
- Re-ranking layer add karna jo teeno retrievers ka output merge kare

Jab ye MVP chal jaye, humein batayein — phir hum ye teen cheezein add karenge.
