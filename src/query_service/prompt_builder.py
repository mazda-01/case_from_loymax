from typing import Iterable, List

from src.config.config import MAX_CONTEXT_CHARS
from src.query_service.schemas import Chunk


SYSTEM_INSTRUCTION = '''
Ты ассистент, который отвечает на вопросы по русскоязычной Википедии.
Используй только приведённые ниже фрагменты текста.
Если ответа в них нет, честно скажи об этом и не придумывай факты.
Отвечай только на русском языке.
'''


def _format_chunk(idx: int, chunk: Chunk) -> str:
    '''Форматирует один чанк в строку с uid, pageid, score и text.'''
    score = "n/a" if chunk.score is None else f"{chunk.score:.4f}"
    header = f"[{idx}] uid={chunk.uid}, ru_wiki_pageid={chunk.ru_wiki_pageid}, score={score}"
    text = chunk.text or ""
    return f"{header}\n{text.strip()}"


def build_prompt(
    question: str,
    chunks: Iterable[Chunk],
    max_chars: int | None = None,
) -> str:
    '''
    Собирает промпт для LLM из вопроса и списка чанков.

    :param question: исходный вопрос пользователя.
    :param chunks: список релевантных чанков (в порядке убывания релевантности).
    :param max_chars: максимальная длина всей текстовой части промпта.
    '''    
    if max_chars is None:
        max_chars = MAX_CONTEXT_CHARS

    parts: List[str] = []
    parts.append(SYSTEM_INSTRUCTION)
    parts.append('')
    parts.append(f'Вопрос пользователя:\n{question.strip()}')
    parts.append('')
    parts.append('Фрагменты контекста (из русской Википедии):')

    current_len = sum(len(p) for p in parts)

    for idx, chunk in enumerate(chunks, start=1):
        formatted = _format_chunk(idx, chunk)
        candidate_len = current_len + 2 + len(formatted)
        if candidate_len > max_chars:
            break
        parts.append('')
        parts.append(formatted)
        current_len = candidate_len

    return '\n'.join(parts)

