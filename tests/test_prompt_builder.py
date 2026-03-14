from src.query_service.schemas import Chunk
from src.query_service.prompt_builder import build_prompt, _format_chunk


def test_format_chunk_structure():
    '''Проверяет структуру строки, возвращаемой _format_chunk (header с uid, pageid, score и text).'''
    chunk = Chunk(uid=1, ru_wiki_pageid=42, score=0.1234, text=" Hello ")
    formatted = _format_chunk(1, chunk)
    assert formatted.startswith("[1] uid=1, ru_wiki_pageid=42, score=0.1234")
    assert "Hello" in formatted


def test_build_prompt_includes_question_and_chunks_and_respects_max_chars():
    '''Проверяет, что промпт содержит вопрос и чанки и обрезается по max_chars.'''
    chunks = [
        Chunk(uid=i, ru_wiki_pageid=100 + i, score=0.9, text=f"text {i}")
        for i in range(3)
    ]
    question = "Кто такой Юрий Гагарин?"

    base_prompt = build_prompt(question, [], max_chars=10_000)
    first_chunk_len = len(_format_chunk(1, chunks[0])) + 2
    max_chars = len(base_prompt) + first_chunk_len
    prompt = build_prompt(question, chunks, max_chars=max_chars)

    assert "Кто такой Юрий Гагарин?" in prompt
    assert "[1] uid=0" in prompt
    assert "[3] uid=2" not in prompt

