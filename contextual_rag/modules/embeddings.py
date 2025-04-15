from langchain_cohere import CohereEmbeddings
from langchain.chat_models import ChatCohere
from config import COHERE_API_KEY

def init_embeddings():
    """Initialize Cohere embeddings"""
    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=COHERE_API_KEY
    )

def init_llm():
    """Initialize Cohere LLM"""
    return ChatCohere(
        model="command",
        temperature=0,
        cohere_api_key=COHERE_API_KEY
    )