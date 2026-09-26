"""
DAY 2: PROCESSING + INDEXING
------------------------------
1. CHUNKING  : lambay documents ko chote pieces mein todna, taake LLM ke
               context window mein sirf relevant chunk fit ho, pura
               document nahi.
2. EMBEDDING : har chunk ko numbers ki list (vector) mein convert karna,
               taake "semantic similarity" se search ho sake (sirf keyword
               match nahi, matlab/meaning se search).
3. INDEXING  : sab vectors ko FAISS mein store karna - ye ek "vector
               database" hai jo fast similarity search allow karta hai.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

VECTORSTORE_DIR = "data/vectorstore"

# Chunking settings explained:
#   chunk_size    -> har chunk mein roughly kitne characters hon
#   chunk_overlap -> chunks ke beech thora overlap, taake context na tootay
#                    (e.g. ek sentence do chunks ke beech na kat jaye)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Free, local embedding model - koi API cost nahi lagti isme
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def chunk_documents(documents: list[Document]) -> list[Document]:
    chunks = text_splitter.split_documents(documents)
    print(f"[indexing] {len(documents)} documents -> {len(chunks)} chunks")
    return chunks


def build_vector_index(documents: list[Document], save_path: str = VECTORSTORE_DIR) -> FAISS:
    """Chunk karta hai, embed karta hai, aur FAISS index disk pe save karta hai."""
    chunks = chunk_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(save_path)
    print(f"[indexing] FAISS index saved to '{save_path}'")
    return vectorstore


def load_vector_index(save_path: str = VECTORSTORE_DIR) -> FAISS:
    """Pehle se saved FAISS index ko wapis load karta hai (dubara embed karne ki zaroorat nahi)."""
    return FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)
