"""
DAY 3 (Part A): SENTENCE-WINDOW RETRIEVER
--------------------------------------------
Idea: Har sentence ko alag se embed karo (bohat precise matching), lekin
retrieval ke waqt sirf wo akela sentence mat do - uske "window" (aas paas
ke N sentences) bhi sath do, taake LLM ko poora context mile.

Example: agar document mein ye 3 sentences hain:
  [0] "The company was founded in 2015."
  [1] "Revenue grew 40% in the first year."
  [2] "This was mainly due to enterprise clients."

Agar user "revenue growth" pooche, match hoga sentence [1]. Lekin sirf
[1] dena adhoora hoga - hum [0], [1], [2] teeno context ke tor pe denge.
"""

import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

SENTENCE_WINDOW_DIR = "data/sentence_window_index"
WINDOW_SIZE = 2  # har taraf kitne sentences include karne hain

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter (punctuation ke basis pe). Chota/production-grade
    use-case ke liye nltk ya spacy behtar hoti hain, lekin ye kaafi hai humare liye."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def build_sentence_window_index(documents: list[Document], save_path: str = SENTENCE_WINDOW_DIR) -> FAISS:
    """
    Har document ko sentences mein todta hai. Har sentence ek alag Document
    ban jata hai, lekin metadata mein hum poori "sentence list" aur is
    sentence ka "index" store karte hain - taake baad mein window nikal sakein.
    """
    sentence_docs = []

    for doc in documents:
        sentences = split_into_sentences(doc.page_content)
        for i, sentence in enumerate(sentences):
            sentence_docs.append(Document(
                page_content=sentence,
                metadata={
                    **doc.metadata,
                    "all_sentences": sentences,   # poora document sentences ki list mein
                    "sentence_index": i,           # ye sentence kis number pe hai
                }
            ))

    vectorstore = FAISS.from_documents(sentence_docs, embedding_model)
    vectorstore.save_local(save_path)
    print(f"[sentence-window] {len(sentence_docs)} sentences indexed -> saved to '{save_path}'")
    return vectorstore


def load_sentence_window_index(save_path: str = SENTENCE_WINDOW_DIR) -> FAISS:
    return FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)


def retrieve_with_window(query: str, top_k: int = 3, window_size: int = WINDOW_SIZE) -> list[dict]:
    """
    Query se sabse milta julta sentence dhoondta hai, phir uske aas paas ke
    sentences jorh kar ek "windowed" context banata hai.
    """
    vectorstore = load_sentence_window_index()
    results = vectorstore.similarity_search(query, k=top_k)

    windowed_results = []
    for doc in results:
        all_sentences = doc.metadata["all_sentences"]
        idx = doc.metadata["sentence_index"]

        start = max(0, idx - window_size)
        end = min(len(all_sentences), idx + window_size + 1)
        window_text = " ".join(all_sentences[start:end])

        windowed_results.append({
            "text": window_text,
            "matched_sentence": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
        })

    return windowed_results