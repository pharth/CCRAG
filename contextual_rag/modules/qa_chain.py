from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Optional

class ContextualQAChain:
    """Question-answering chain for Contextual RAG"""
    
    def __init__(self, vector_store: Chroma, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Query reformulation prompt
        self.query_reformulation_prompt = ChatPromptTemplate.from_template("""
        Given the conversation history and the current question, generate a search query 
        that captures the full context needed to answer the current question accurately.
        
        Conversation History:
        {history}
        
        Current Question: {question}
        
        Reformulated Search Query:
        """)
        
        # Answer generation prompt
        self.answer_prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that provides accurate information based on the context provided.
        
        Answer the question based on the following context and conversation history:
        
        Context:
        {context}
        
        Conversation History:
        {history}
        
        Question: {question}
        
        Provide a comprehensive and accurate answer using only the information in the context.
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
    
    def _get_context(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Get context for query"""
        reformulated_query = self.reformulate_query(query, history)
        docs = self.retriever.get_relevant_documents(reformulated_query)
        
        # Extract original content from metadata if available
        context_texts = []
        for doc in docs:
            if "original_content" in doc.metadata:
                # Using the original content, not the contextualized version
                # We already leveraged the context for better retrieval
                context_texts.append(doc.metadata["original_content"])
            else:
                context_texts.append(doc.page_content)
        
        return "\n\n".join(context_texts)
    
    def generate_answer(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate answer to query using contextual RAG"""
        if not history:
            history = []
            
        # Get context
        context = self._get_context(query, history)
        
        # Format history
        formatted_history = self._format_history(history)
        
        # Generate answer
        answer_chain = (
            self.answer_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return answer_chain.invoke({
            "context": context,
            "history": formatted_history,
            "question": query
        })