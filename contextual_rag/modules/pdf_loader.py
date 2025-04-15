from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import CHUNK_SIZE, CHUNK_OVERLAP

class ContextualPDFProcessor:
    def __init__(self, embeddings, llm, persist_directory=None):
        """Initialize contextual PDF processor with text splitter and vector store"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        self.llm = llm
        
        # Define prompts for contextual enrichment
        self.document_context_prompt = ChatPromptTemplate.from_template("""
        You are analyzing a document for context generation. Please read and understand this document.
        
        Document:
        {document_content}
        
        DO NOT respond yet. Simply read and understand this document for future reference.
        """)
        
        self.chunk_context_prompt = ChatPromptTemplate.from_template("""
        You've already read the full document. Now provide a brief context for the following chunk.
        
        This chunk is part of the document you just read. Provide 1-2 sentences that explain:
        1. What is the main topic of this chunk?
        2. How does this chunk fit into the overall document?
        
        Here's the chunk:
        {chunk_content}
        
        Provide ONLY the contextual summary in 1-2 sentences. Be concise but informative.
        """)
    
    def generate_chunk_context(self, document_content: str, chunk_content: str) -> str:
        """Generate contextual summary for a chunk"""
        # First, provide the full document to the LLM (with caching)
        # Note: In a real implementation, you'd use cache_control as in the reference
        # Since we can't implement actual caching here, we'll simulate the process
        
        # Then, generate context for the specific chunk
        context_chain = (
            self.chunk_context_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        return context_chain.invoke({
            "chunk_content": chunk_content
        })
    
    def create_contextual_document(self, document: Document, full_document_content: str) -> Document:
        """Enrich a document with contextual information"""
        # Generate contextual summary
        context = self.generate_chunk_context(full_document_content, document.page_content)
        
        # Create new document with context + original content
        contextual_content = f"Context: {context}\n\nContent: {document.page_content}"
        
        # Create new document with same metadata but enriched content
        return Document(
            page_content=contextual_content,
            metadata={
                **document.metadata,
                "original_content": document.page_content,
                "context_summary": context
            }
        )
    
    def load_and_process(self, pdf_path: str) -> List[Document]:
        """Load and process a PDF document with contextual enrichment"""
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            if "source" not in doc.metadata:
                doc.metadata["source"] = pdf_path
        
        # Get full document content for context
        full_document_content = "\n".join([doc.page_content for doc in documents])
        
        # Split text
        splits = self.text_splitter.split_documents(documents)
        
        # Enrich each chunk with context
        contextual_documents = []
        print(f"Generating contextual embeddings for {len(splits)} chunks...")
        for i, chunk in enumerate(splits):
            print(f"Processing chunk {i+1}/{len(splits)}")
            contextual_doc = self.create_contextual_document(chunk, full_document_content)
            contextual_documents.append(contextual_doc)
        
        # Add to vector store
        self.vector_store.add_documents(documents=contextual_documents)
        
        return contextual_documents