import argparse
import logging

from src.config.config import COLLECTION_NAME
from src.indexing_service.pipeline import run_indexing


def parse_args() -> argparse.Namespace:
    '''Разбор аргументов CLI для indexing service (source, collection-name, parquet-output, batch-size).'''
    parser = argparse.ArgumentParser(
        description='Indexing Service: load > quality checks > preprocess > vectorize > Qdrant',
    )

    parser.add_argument(
        '--source',
        type=str,
        default=None,
        help='Путь или URL к сырым данным (JSON / parquet). '
        'Если не указан, используется дефолтный URL.',
    )
    parser.add_argument(
        '--collection-name',
        type=str,
        default=COLLECTION_NAME,
        help='Имя коллекции в Qdrant (по умолчанию из config.COLLECTION_NAME).',
    )
    parser.add_argument(
        '--parquet-output',
        type=str,
        default='data/rubq_paragraphs.parquet',
        help='Путь для сохранения препроцессированных данных в parquet.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Размер батча для векторизации и upsert в Qdrant.',
    )

    return parser.parse_args()


def main() -> None:
    '''Точка входа: настройка логирования и вызов run_indexing с аргументами CLI.'''

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    args = parse_args()

    parquet_output = args.parquet_output if args.parquet_output else None

    run_indexing(
        source=args.source,
        collection_name=args.collection_name,
        parquet_output=parquet_output,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()

