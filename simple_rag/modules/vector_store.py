from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document

class VectorStore:
    """Vector store for document embeddings"""
    
    def __init__(self, embeddings, persist_directory=None):
        self.store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        self.store.add_documents(documents=documents)
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Perform similarity search on the vector store"""
        return self.store.similarity_search(query, k=k)
    
    def as_retriever(self, search_kwargs=None):
        """Return the vector store as a retriever"""
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        return self.store.as_retriever(search_kwargs=search_kwargs)