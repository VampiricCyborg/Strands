# rag_index.py
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


DOCS_DIR = "data/docs"           # where your documents live
VECTOR_DIR = "data/vector_store" # where Chroma will persist
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents():
    docs = []
    docs_path = Path(DOCS_DIR)
    docs_path.mkdir(parents=True, exist_ok=True)

    for file in docs_path.glob("*"):
        print(f"📄 Loading {file.name}")
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix.lower() == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
        else:
            print(f"⚠️ Unsupported file type: {file.suffix}")
            continue

        file_docs = loader.load()
        # keep original filename as metadata
        for d in file_docs:
            d.metadata["source"] = file.name
        docs.extend(file_docs)

    print(f"✅ Loaded {len(docs)} documents/pages")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)
    print(f"🧱 Split into {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    # This overwrites previous index; run again if you change docs
    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DIR,
    )
    db.persist()
    print(f"💾 Vector store saved to: {VECTOR_DIR}")


if __name__ == "__main__":
    documents = load_documents()
    if not documents:
        print("⚠️ No documents found in data/docs. Add some PDFs/TXTs and rerun.")
    else:
        chunks = split_documents(documents)
        build_vector_store(chunks)
