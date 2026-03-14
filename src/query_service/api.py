import logging

from fastapi import FastAPI, HTTPException

from src.query_service.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
app = FastAPI(title='RAG - Query Service')

_embedding_model_ready = False


def _load_embedding_model() -> None:
    '''Загружает модель эмбеддингов и прогревает ее. Вызывается при startup.'''
    global _embedding_model_ready
    try:
        logger.info('Loading embedding model...')
        from src.embeddings import model as emb_model

        emb_model.model.encode(['warmup'], show_progress_bar=False)
        _embedding_model_ready = True
        logger.info('Embedding model loaded')
    except Exception as exc:
        logger.exception('Failed to load embedding model: %s', exc)
        _embedding_model_ready = False


@app.on_event('startup')
def startup():
    '''FastAPI startup: загрузка модели эмбеддингов.'''

    _load_embedding_model()
    logger.info('Query service ready (startup complete)')


@app.get('/health')
def health() -> dict:
    '''Эндпоинт для проверки, что сервис поднят (без загрузки RAG).'''
    logger.info('GET /health')
    return {"status": "ok"}


@app.get('/ready')
def ready() -> dict:
    '''Возвращает 200, когда модель эмбеддингов загружена и сервис готов к /query.'''
    if _embedding_model_ready:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail='Embedding model not loaded')


@app.post('/query', response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    '''Обработка POST /query: проверка готовности, вызов run_rag, возврат QueryResponse.'''
    if not _embedding_model_ready:
        raise HTTPException(
            status_code=503, detail='Embedding model not loaded'
        )
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail='Question must not be empty')

    from src.query_service.rag_chain import run_rag

    try:
        answer, chunks = run_rag(question, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return QueryResponse(question=question, answer=answer, chunks=chunks)

