import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Vector database settings
PERSIST_DIRECTORY = "vector_db"

# Text chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Default PDF directory
DEFAULT_PDF_DIR = "data/mirage"

# Retrieval parameters
DEFAULT_RETRIEVAL_K = 3