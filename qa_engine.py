"""
QUERY CONSTRUCTION + LLM CALL + CITATIONS  (with conversation memory + chunk transparency + full-CSV fallback)
--------------------------------------------------------------------------------------------------------------
Naya: agar chota CSV dataset index hua hai, uska POORA data bhi context
mein add hota hai - taake "filter/compare across all rows" type sawalon
(jaise "7 seats wali konsi car hai") ka sahi jawab mile, sirf top-k
retrieval pe depend na rahe.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from multi_retriever import multi_retrieve

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

SYSTEM_PROMPT = """You are a knowledge-base assistant. Answer ONLY using the
provided context. If the answer is not in the context, say you don't know -
never make something up. Use the conversation history to understand
follow-up questions (e.g. "explain the first one" refers to something
mentioned earlier), but still ground your answer in the provided context.
Keep answers concise.

IMPORTANT: Always answer in the SAME language the user's question is written
in. If the question is in English, answer in English. If the question is in
Urdu (or Roman Urdu, i.e. Urdu written in English letters), answer in that
same style. Match the user's language exactly - do not switch languages.

If a "Full dataset" section is present in the context, use it to answer
questions that require checking, filtering, or comparing across ALL rows
(e.g. "which has the most seats", "is there a 2019 model") - don't rely
only on the top retrieved snippets for such questions."""

MAX_HISTORY_TURNS = 3
CSV_DUMP_PATH = "data/full_csv_dump.txt"


def answer_question(query: str, history: list[dict] | None = None, final_k: int = 4) -> dict:
    results = multi_retrieve(query, top_k_per_retriever=4, final_k=final_k)

    if not results:
        return {
            "answer": "Mujhe is sawal ka jawab apne documents mein nahi mila.",
            "sources": [], "retrievers_used": [], "retrieved_chunks": [],
        }

    context_parts, sources, retrievers_used, retrieved_chunks = [], [], [], []

    for r in results:
        context_parts.append(r["text"])
        source_label = r["source"]
        page = r.get("page")
        sources.append(f"{source_label}" + (f" (page {page})" if page is not None else ""))
        retrievers_used.append(r["retriever"])
        retrieved_chunks.append({
            "text": r["text"],
            "source": source_label,
            "retriever": r["retriever"],
            "score": round(r.get("score", 0), 3),
        })

    context = "\n\n---\n\n".join(context_parts)

    # Chota CSV dataset ho to poora data bhi context mein shamil karo
    if os.path.exists(CSV_DUMP_PATH):
        with open(CSV_DUMP_PATH, "r", encoding="utf-8") as f:
            full_csv_text = f.read()
        context += f"\n\n---\n\nFull dataset (all rows, for accurate filtering/comparison):\n{full_csv_text}"

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        trimmed = history[-(MAX_HISTORY_TURNS * 2):]
        messages.extend(trimmed)
    messages.append({"role": "user", "content": user_prompt})

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": list(dict.fromkeys(sources)),
        "retrievers_used": list(dict.fromkeys(retrievers_used)),
        "retrieved_chunks": retrieved_chunks,
    }


if __name__ == "__main__":
    result = answer_question("What is this document about?")
    print(result["answer"])