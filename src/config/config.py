import os

from dotenv import load_dotenv


load_dotenv()

EMBEDDING_MODEL = 'sentence-transformers/LaBSE'
QDRANT_URL = 'http://localhost:6333'
VECTOR_SIZE = 768
COLLECTION_NAME = 'rubq_paragraphs'

HF_API_TOKEN = os.getenv('HF_API_TOKEN')
HF_LLM_MODEL = os.getenv('HF_LLM_MODEL')
HF_LLM_MAX_NEW_TOKENS = 1024

MAX_CONTEXT_CHARS = 8000
RETRIEVAL_SCORE_THRESHOLD = 0.5