from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ContextualRetriever(BaseRetriever):
    """Contextual retriever for RAG"""
    
    def __init__(self, vector_store: Chroma, llm, k: int = 3):
        super().__init__()
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
        
        # Query reformulation prompt
        self.query_reformulation_prompt = ChatPromptTemplate.from_template("""
        Given the conversation history and the current question, generate a search query 
        that captures the full context needed to answer the current question accurately.
        
        Conversation History:
        {history}
        
        Current Question: {question}
        
        Reformulated Search Query:
        """)
    
    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history"""
        if not history:
            return "No prior conversation."
        
        formatted_history = []
        for entry in history:
            role = entry.get("role", "unknown")
            content = entry.get("content", "")
            formatted_history.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_history)
    
    def reformulate_query(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Reformulate query based on conversation history"""
        if not history:
            return query
        
        formatted_history = self._format_history(history)
        
        reformulation_chain = (
            self.query_reformulation_prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            return reformulation_chain.invoke({
                "history": formatted_history,
                "question": query
            })
        except Exception as e:
            print(f"Error in query reformulation: {e}")
            return query
    
    def _get_relevant_documents(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> List[Document]:
        """Get relevant documents for a query with conversation history"""
        reformulated_query = self.reformulate_query(query, history)
        return self.vector_store.similarity_search(reformulated_query, k=self.k)
    
    def get_relevant_documents(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> List[Document]:
        """Public method to get relevant documents for a query with conversation history"""
        return self._get_relevant_documents(query, history)