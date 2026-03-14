from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pandas as pd
from typing import Iterable, List
import hashlib

from src.embeddings.model import model
from src.config.config import QDRANT_URL, VECTOR_SIZE, COLLECTION_NAME

def get_client() -> QdrantClient:
    '''Возвращает клиент Qdrant.'''
    return QdrantClient(url=QDRANT_URL, prefer_grpc=True)


def init_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME):
    '''Создаёт коллекцию в Qdrant, если не существует.'''

    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

def encode_texts(texts: Iterable[str], batch_size: int = 64) -> List[List[float]]:
    '''Батчевая векторизация текстов; возвращает список векторов.'''

    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def _to_point_id(source_uid: int | str, row_index: int) -> int:
    '''Стабильный целочисленный id точки из source_uid и row_index для Qdrant.'''

    raw = f"{source_uid}:{row_index}".encode('utf-8') # Однозначно переводим в байты
    return int.from_bytes(hashlib.sha1(raw).digest()[:8], 'big') & ((1 << 63) - 1) # Диапазон от единицы до 63 бита


def vectorize_documents(
    df: pd.DataFrame,
    text_column: str = 'text',
    id_column: str = 'uid',
    pageid_column: str = 'ru_wiki_pageid',
    batch_size: int = 256,
) -> List[PointStruct]:
    '''По DataFrame строит список PointStruct (id, vector, payload: uid, ru_wiki_pageid, text).'''

    texts = df[text_column].tolist()
    embeddings = encode_texts(texts, batch_size=batch_size)

    points: List[PointStruct] = []
    for row, vector in zip(df.itertuples(index=True), embeddings):
        row_dict = row._asdict()
        source_uid = row_dict[id_column]
        point_id = _to_point_id(source_uid, row_dict['Index'])

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    'uid': row_dict[id_column],
                    'ru_wiki_pageid': row_dict[pageid_column],
                    'text': row_dict[text_column],
                },
            )
        )
    return points

def index_dataframe(
    df: pd.DataFrame,
    text_column: str = 'text',
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 256,
    id_column: str = 'uid',
    pageid_column: str = 'ru_wiki_pageid',
) -> None:
    '''Инициализация коллекции и upsert чанков из DataFrame батчами.'''

    client = get_client()
    init_collection(client, collection_name)
    n = len(df)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_df = df.iloc[start:end]
        points = vectorize_documents(
            df=batch_df,
            text_column=text_column,
            id_column=id_column,
            pageid_column=pageid_column,
            batch_size=batch_size,
        )
        client.upsert(collection_name=collection_name, points=points, wait=False)

if __name__ == '__main__':
    df = pd.read_parquet('data/rubq_paragraphs.parquet')
    index_dataframe(df)