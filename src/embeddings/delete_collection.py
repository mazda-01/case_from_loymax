'''Утилита удаления коллекции Qdrant для разработки.'''

from qdrant_client import QdrantClient
from src.config.config import QDRANT_URL, COLLECTION_NAME

if __name__ == '__main__':
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f'Collection "{COLLECTION_NAME}" deleted')
    else:
        print(f'Collection "{COLLECTION_NAME}" does not exist')