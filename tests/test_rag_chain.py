from types import SimpleNamespace

import pytest

import src.query_service.rag_chain as rag_chain


class DummyQdrantClient:
    '''Мок Qdrant-клиента для тестов retrieve_chunks.'''

    def search(self, collection_name, query_vector, limit, with_payload):
        assert collection_name
        assert isinstance(query_vector, list)
        assert limit == 2
        assert with_payload is True
        return [
            SimpleNamespace(
                score=0.91,
                payload={"uid": 1, "ru_wiki_pageid": 10, "text": "doc1"},
            ),
            SimpleNamespace(
                score=0.77,
                payload={"uid": 2, "ru_wiki_pageid": 20, "text": "doc2"},
            ),
        ]


def test_retrieve_chunks_reads_payload_and_score(monkeypatch: pytest.MonkeyPatch):
    '''Проверяет, что retrieve_chunks возвращает чанки с uid, ru_wiki_pageid, text и score из payload.'''
    monkeypatch.setattr(rag_chain, "get_qdrant_client", lambda: DummyQdrantClient())
    monkeypatch.setattr(rag_chain, "_embed_query", lambda question: [0.1, 0.2, 0.3])

    chunks = rag_chain.retrieve_chunks("Тестовый вопрос", top_k=2)

    assert len(chunks) == 2
    assert chunks[0].uid == 1
    assert chunks[0].ru_wiki_pageid == 10
    assert chunks[0].text == "doc1"
    assert chunks[0].score == pytest.approx(0.91)


def test_run_rag_calls_llm_with_prompt(monkeypatch: pytest.MonkeyPatch):
    '''Проверяет, что run_rag вызывает LLM с промптом и возвращает ответ и чанки.'''
    fake_chunks = [
        rag_chain.Chunk(uid=1, ru_wiki_pageid=10, score=0.9, text="ctx1"),
    ]
    monkeypatch.setattr(rag_chain, "retrieve_chunks", lambda question, top_k: fake_chunks)
    monkeypatch.setattr(rag_chain, "build_prompt", lambda question, chunks: "PROMPT")
    monkeypatch.setattr(
        rag_chain,
        "generate_answer",
        lambda prompt, max_new_tokens=None: "MOCK_ANSWER",
    )

    answer, chunks = rag_chain.run_rag("Вопрос", top_k=1)

    assert answer == "MOCK_ANSWER"
    assert chunks == fake_chunks


def test_run_rag_raises_on_empty_question():
    '''Проверяет, что run_rag бросает ValueError при пустом вопросе.'''
    with pytest.raises(ValueError):
        rag_chain.run_rag("   ", top_k=1)


def test_run_rag_returns_no_data_when_all_chunks_below_threshold(monkeypatch):
    '''Если все чанки ниже порога релевантности, возвращаем сообщение об отсутствии данных и пустой список.'''
    low_score_chunks = [
        rag_chain.Chunk(uid=1, ru_wiki_pageid=10, score=0.3, text="irrelevant"),
        rag_chain.Chunk(uid=2, ru_wiki_pageid=20, score=0.25, text="also irrelevant"),
    ]
    monkeypatch.setattr(rag_chain, "retrieve_chunks", lambda question, top_k: low_score_chunks)

    answer, chunks = rag_chain.run_rag("Случайный запрос", top_k=2)

    assert answer == rag_chain.NO_RELEVANT_DATA_ANSWER
    assert chunks == []

