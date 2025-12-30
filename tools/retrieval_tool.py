# rag_tools.py
from strands import tool  
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


VECTOR_DIR = "data/vector_store"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# global, so it's loaded once
_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
_vectorstore = Chroma(
    persist_directory=VECTOR_DIR,
    embedding_function=_embeddings,
)


@tool
def retrieve_docs(query: str) -> str:
    """
    Search the indexed documents and return the most relevant passages
    for the given query. Used for Retrieval-Augmented Generation (RAG).
    """
    try:
        docs = _vectorstore.similarity_search(query, k=4)
    except Exception as e:
        return f"Error querying vector store: {e}"

    if not docs:
        return "No relevant context found in the indexed documents."

    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:400].replace("\n", " ")
        parts.append(f"[{i}] Source: {source}\n{snippet}")

    return "\n\n".join(parts)
