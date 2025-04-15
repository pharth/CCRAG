from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document

class SimpleRetriever(BaseRetriever):
    """Simple retriever for RAG"""
    
    def __init__(self, vector_store: Chroma, k: int = 3):
        super().__init__()
        self.vector_store = vector_store
        self.k = k
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query"""
        return self.vector_store.similarity_search(query, k=self.k)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Public method to get relevant documents for a query"""
        return self._get_relevant_documents(query)