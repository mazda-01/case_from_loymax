from typing import List

from qdrant_client import QdrantClient

from src.config.config import (
    COLLECTION_NAME,
    HF_LLM_MAX_NEW_TOKENS,
    QDRANT_URL,
    RETRIEVAL_SCORE_THRESHOLD,
)
from src.embeddings.model import model as sentence_model
from src.query_service.llm_client import generate_answer
from src.query_service.prompt_builder import build_prompt
from src.query_service.schemas import Chunk


def get_qdrant_client() -> QdrantClient:
    '''Создаёт и возвращает клиент Qdrant по QDRANT_URL.'''
    return QdrantClient(url=QDRANT_URL)


def _embed_query(question: str) -> List[float]:
    '''Эмбеддинг вопроса через sentence model; возвращает список float.'''
    vector = sentence_model.encode(
        [question],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    return vector.tolist()


def retrieve_chunks(question: str, top_k: int) -> List[Chunk]:
    '''Поиск top_k чанков по вопросу в Qdrant; возвращает список Chunk с payload и score.'''
    if top_k < 1:
        raise ValueError('top_k must be >= 1')

    client = get_qdrant_client()
    query_vector = _embed_query(question)
    points = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )

    chunks: List[Chunk] = []
    for point in points:
        payload = point.payload or {}
        chunks.append(
            Chunk(
                uid=payload.get('uid', ''),
                ru_wiki_pageid=payload.get('ru_wiki_pageid', ''),
                score=float(point.score) if point.score is not None else None,
                text=payload.get('text'),
            )
        )
    return chunks


NO_RELEVANT_DATA_ANSWER = (
    'В базе знаний не найдено достаточно релевантных данных по вашему запросу.'
)


def run_rag(question: str, top_k: int) -> tuple[str, List[Chunk]]:
    '''
    Полный RAG: retrieve > фильтр по порогу > build_prompt > generate_answer. 
    Бросает ValueError при пустом вопросе.
    '''
    clean_question = question.strip()
    if not clean_question:
        raise ValueError('Question must not be empty')

    chunks = retrieve_chunks(clean_question, top_k=top_k)
    relevant = [
        c
        for c in chunks
        if c.score is not None and c.score >= RETRIEVAL_SCORE_THRESHOLD
    ]
    if not relevant:
        return NO_RELEVANT_DATA_ANSWER, []

    prompt = build_prompt(clean_question, relevant)
    answer = generate_answer(prompt, max_new_tokens=HF_LLM_MAX_NEW_TOKENS)
    return answer, relevant

