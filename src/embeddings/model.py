'''Загрузка модели SentenceTransformer.'''

import dotenv
import os
from sentence_transformers import SentenceTransformer
from src.config.config import EMBEDDING_MODEL
import torch

dotenv.load_dotenv()

HF_TOKEN = os.getenv('HF_API_TOKEN')

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = SentenceTransformer(EMBEDDING_MODEL, use_auth_token=HF_TOKEN, device=device)