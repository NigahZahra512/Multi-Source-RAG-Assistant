"""
DAY 3 (Part C): MULTI-RETRIEVER + RE-RANKING
------------------------------------------------
Ab teeno retrievers ko jorte hain:
  1. Vector retriever          (indexing.py)        - general semantic search
  2. Sentence-window retriever (sentence_window_retriever.py) - precise + context
  3. Graph retriever           (graph_retriever.py)  - relationship-based

Har retriever apne candidates deta hai. Phir hum sab ko ek list mein daal
kar "Cross-Encoder" model se RE-RANK karte hain.

Cross-Encoder vs normal embeddings ka farq:
  - Normal embedding: query aur document ko ALAG ALAG embed karta hai,
    phir unki similarity nikalta hai (fast, lekin thora less accurate)
  - Cross-Encoder: query + document DONO ko EK SATH model ko deta hai,
    model seedha ek "relevance score" deta hai (slow hota hai bade scale
    pe, isliye ye sirf FINAL top candidates pe use karte hain - "re-ranking"
    isi liye kehte hain: pehle retrieval se rough candidates lo, phir
    inhe accurately re-rank karo)
"""

from sentence_transformers import CrossEncoder
from indexing import load_vector_index
from sentence_window_retriever import retrieve_with_window
from graph_retriever import retrieve_from_graph

# Ye chota, fast cross-encoder model hai jo query-document pairs ko score karta hai
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def multi_retrieve(query: str, top_k_per_retriever: int = 4, final_k: int = 4) -> list[dict]:
    """
    Teeno retrievers se candidates collect karta hai, phir cross-encoder
    se re-rank kar ke sabse relevant top `final_k` candidates return karta hai.
    """
    candidates = []

    # --- Retriever 1: Vector Search ---
    vectorstore = load_vector_index()
    vector_results = vectorstore.similarity_search(query, k=top_k_per_retriever)
    for doc in vector_results:
        candidates.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "retriever": "vector",
        })

    # --- Retriever 2: Sentence-Window ---
    window_results = retrieve_with_window(query, top_k=top_k_per_retriever)
    for r in window_results:
        candidates.append({
            "text": r["text"],
            "source": r["source"],
            "page": r["page"],
            "retriever": "sentence-window",
        })

    # --- Retriever 3: Graph-Based ---
    try:
        graph_contexts = retrieve_from_graph(query)
        for context in graph_contexts:
            candidates.append({
                "text": context,
                "source": "knowledge-graph",
                "page": None,
                "retriever": "graph",
            })
    except FileNotFoundError:
        pass  # graph abhi build nahi hua, koi baat nahi - baaki retrievers chalenge

    if not candidates:
        return []

    # --- Re-Ranking: Cross-Encoder se sab candidates ko score karo ---
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["score"] = float(score)

    # Highest score wale upar
    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)

    # Duplicate / bohat milte julte text hata do
    seen_texts = set()
    final_results = []
    for c in ranked:
        key = c["text"][:80]  # pehle 80 characters se rough duplicate check
        if key not in seen_texts:
            seen_texts.add(key)
            final_results.append(c)
        if len(final_results) >= final_k:
            break

    return final_results