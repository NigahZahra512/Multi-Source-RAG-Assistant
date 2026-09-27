"""
DAY 1: DATA INGESTION PIPELINE
--------------------------------
Ye file teen alag data sources se text nikal kar ek "Unified Document"
format mein laati hai (LangChain ka Document object).

Har Document mein do cheezein hoti hain:
  - page_content : asal text
  - metadata     : { source, type, ...} -> baad mein citation ke liye use hoga

Isay "Unified Document DB" isliye kehte hain kyunke chahe data PDF se aaye,
website se aaye, ya CSV se aaye - sab ek hi shape mein store hota hai.
Isi wajah se aage chunking/indexing code ko ye farq nahi karna padta ke
data kahan se aaya tha.
"""

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, CSVLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """Ek PDF file se text nikalta hai, page-wise Document objects banata hai."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for d in docs:
        d.metadata["source_type"] = "pdf"
    print(f"[ingestion] {file_path} -> {len(docs)} pages loaded")
    return docs


def load_website(url: str) -> list[Document]:
    """Ek website URL se text nikalta hai."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    for d in docs:
        d.metadata["source_type"] = "website"
    print(f"[ingestion] {url} -> {len(docs)} page(s) loaded")
    return docs


def load_csv(file_path: str) -> list[Document]:
    """CSV file ki har row ko ek Document bana deta hai."""
    loader = CSVLoader(file_path)
    docs = loader.load()
    for d in docs:
        d.metadata["source_type"] = "csv"
    print(f"[ingestion] {file_path} -> {len(docs)} rows loaded")
    return docs


def build_unified_documents(pdf_paths=None, urls=None, csv_paths=None) -> list[Document]:
    """
    Teeno sources se data le kar ek single list mein jorta hai.
    Ye function hi "Unified Document DB" hai (Step 3 of the diagram).
    """
    all_docs: list[Document] = []

    for path in (pdf_paths or []):
        all_docs.extend(load_pdf(path))

    for url in (urls or []):
        all_docs.extend(load_website(url))

    for path in (csv_paths or []):
        all_docs.extend(load_csv(path))

    print(f"[ingestion] TOTAL unified documents: {len(all_docs)}")
    return all_docs


def save_full_csv_dump(documents: list[Document], threshold: int = 50, save_path: str = "data/full_csv_dump.txt"):
    """
    Agar CSV rows ki tadaad chhoti hai (<= threshold), to poora CSV data
    ek text file mein save kar dete hain. Ye baad mein qa_engine.py mein
    HAR sawal ke context mein shamil hoga - taake "filter across all rows"
    type sawalon (jaise "7 seats wali konsi car hai") ka SAHI jawab mile,
    sirf top-k similarity search pe depend na rahe.
    """
    csv_docs = [d for d in documents if d.metadata.get("source_type") == "csv"]

    if not csv_docs or len(csv_docs) > threshold:
        print(f"[csv-dump] Skipped - either no CSV or too many rows ({len(csv_docs)})")
        return

    with open(save_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(csv_docs):
            f.write(f"Row {i+1}: {doc.page_content}\n")

    print(f"[csv-dump] Saved {len(csv_docs)} CSV rows in full to '{save_path}'")


if __name__ == "__main__":
    docs = build_unified_documents(
        pdf_paths=[],
        urls=[],
        csv_paths=[],
    )
    for d in docs[:2]:
        print("---")
        print(d.metadata)
        print(d.page_content[:200])