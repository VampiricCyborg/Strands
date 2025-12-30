import chromadb

class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="rag_vectorstore")
        self.collection = self.client.get_or_create_collection("rag_docs")

    def add(self, id, content, metadata=None):
        self.collection.add(
            ids=[id],
            documents=[content],
            metadatas=[metadata]
        )

    def search(self, query, top_k=3):
        return self.collection.query(query_texts=[query], n_results=top_k)
