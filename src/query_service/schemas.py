from typing import List

from pydantic import BaseModel


class QueryRequest(BaseModel):
    '''Тело запроса к /query: вопрос и top_k.'''

    question: str
    top_k: int = 5


class Chunk(BaseModel):
    '''Один релевантный фрагмент: uid, ru_wiki_pageid, score, text.'''

    uid: int | str
    ru_wiki_pageid: int | str
    score: float | None = None
    text: str | None = None


class QueryResponse(BaseModel):
    '''Ответ /query: question, answer, chunks.'''

    question: str
    answer: str
    chunks: List[Chunk]
