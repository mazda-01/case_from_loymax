from fastapi.testclient import TestClient

import src.query_service.api as api_module
from src.query_service.api import app


def test_query_endpoint_with_mocked_rag_chain(monkeypatch):
    '''Проверяет, что POST /query возвращает 200 и ответ с question, answer, chunks.'''

    monkeypatch.setattr(api_module, "_embedding_model_ready", True)
    mock_run_rag = lambda question, top_k: (
        "MOCK_ANSWER",
        [
            {
                "uid": 1,
                "ru_wiki_pageid": 42,
                "score": 0.8,
                "text": "context text",
            }
        ],
    )
    monkeypatch.setattr(
        'src.query_service.rag_chain.run_rag',
        mock_run_rag,
    )

    client = TestClient(app)

    resp = client.post(
        '/query',
        json={"question": "Кто такой Юрий Гагарин?", "top_k": 3},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["question"]
    assert data["answer"] == "MOCK_ANSWER"
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["uid"] == 1
    assert data["chunks"][0]["ru_wiki_pageid"] == 42


def test_ready_returns_503_when_model_not_loaded(monkeypatch):
    '''Проверяет, что GET /ready возвращает 503, когда модель не загружена.'''
    monkeypatch.setattr(api_module, "_embedding_model_ready", False)
    client = TestClient(app)
    resp = client.get("/ready")
    assert resp.status_code == 503


def test_ready_returns_200_when_model_loaded(monkeypatch):
    '''Проверяет, что GET /ready возвращает 200 и status=ready при загруженной модели.'''
    monkeypatch.setattr(api_module, "_embedding_model_ready", True)
    client = TestClient(app)
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


def test_query_returns_503_when_model_not_loaded(monkeypatch):
    '''Проверяет, что POST /query возвращает 503 с detail о незагруженной модели.'''
    monkeypatch.setattr(api_module, "_embedding_model_ready", False)
    client = TestClient(app)
    resp = client.post("/query", json={"question": "test?", "top_k": 3})
    assert resp.status_code == 503
    assert "not loaded" in resp.json()["detail"].lower()

