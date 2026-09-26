"""
QUERY CONSTRUCTION + LLM CALL + CITATIONS
--------------------------------------------
Ye file diagram ke steps 6-9 cover karti hai (abhi ke liye single
vector-retriever ke sath - multi-retriever + re-ranking hum Day 3 mein
add karenge).

Flow:
  1. User ka sawal aata hai
  2. Retriever FAISS se top-k relevant chunks nikalta hai   (Retrieval)
  3. Un chunks ko ek prompt mein LLM ko diya jata hai        (Query Construction)
  4. LLM answer generate karta hai, sources ke sath           (Response Format)
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from indexing import load_vector_index

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

SYSTEM_PROMPT = """You are a knowledge-base assistant. Answer ONLY using the
provided context. If the answer is not in the context, say you don't know -
never make something up. Keep answers concise. After the answer, list the
sources you used."""


def answer_question(query: str, top_k: int = 4) -> dict:
    vectorstore = load_vector_index()

    # Retrieval: FAISS se sab se relevant chunks nikalna
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    context_parts = []
    sources = []
    for doc, score in results:
        context_parts.append(doc.page_content)
        source_label = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        sources.append(f"{source_label}" + (f" (page {page})" if page is not None else ""))

    context = "\n\n---\n\n".join(context_parts)

    # Query Construction: context + question ko ek prompt mein jorna
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    return {
        "answer": response.content,
        "sources": list(dict.fromkeys(sources)),  # duplicates hata dega, order rakhega
    }


if __name__ == "__main__":
    result = answer_question("What is this document about?")
    print(result["answer"])
    print("\nSources:", result["sources"])
