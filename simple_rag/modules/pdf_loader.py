from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP

class PDFProcessor:
    def __init__(self, embeddings, persist_directory=None):
        """Initialize PDF processor with text splitter and vector store"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
    
    def load_and_process(self, pdf_path: str) -> List[Document]:
        """Load and process a PDF document"""
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            if "source" not in doc.metadata:
                doc.metadata["source"] = pdf_path
        
        # Split text
        splits = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(documents=splits)
        
        return splits