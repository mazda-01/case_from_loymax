import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.loader import load_data, URL as DEFAULT_URL
from src.data.preprocessing import run_quality_checks, preprocess_data
from src.embeddings.vector_store import index_dataframe
from src.config.config import COLLECTION_NAME


logger = logging.getLogger(__name__)


def load_raw(source: Optional[str]) -> pd.DataFrame:
    '''Загрузка сырых данных: URL JSON, путь к parquet или default URL > возвращает DataFrame.'''

    if not source:
        logger.info('Loading data from default URL: %s', DEFAULT_URL)
        return load_data(DEFAULT_URL)

    if source.startswith('http://') or source.startswith('https://') or source.endswith('.json'):
        logger.info('Loading raw JSON data from %s', source)
        return load_data(source)

    path = Path(source)
    if path.suffix == '.parquet':
        logger.info('Loading preprocessed data from parquet: %s', source)
        return pd.read_parquet(path)

    raise ValueError(f'Unsupported source format: {source}')


def run_indexing(
    source: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
    parquet_output: Optional[str] = 'data/rubq_paragraphs.parquet',
    batch_size: int = 256,
) -> None:
    '''Пайплайн: load_raw > quality_checks > preprocess > сохранение parquet (опционально) > index_dataframe.'''

    logger.info('Starting indexing pipeline')

    df = load_raw(source)
    logger.info('Loaded raw data: %d rows', len(df))

    df = run_quality_checks(df)
    logger.info('After quality checks: %d rows"', len(df))

    df = preprocess_data(df)
    logger.info('After preprocessing / chunking: %d rows', len(df))

    if parquet_output:
        output_path = Path(parquet_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        logger.info('Saved preprocessed data to %s', output_path)

    logger.info(
        'Indexing into Qdrant collection "%s" with batch_size=%d',
        collection_name,
        batch_size,
    )
    index_dataframe(
        df,
        collection_name=collection_name,
        batch_size=batch_size,
    )
    logger.info('Indexing pipeline completed')


