'''Логирование quality checks и препроцессинг/чанкинг для RuBQ.'''

import pandas as pd
import logging
from .loader import load_data, URL

logger = logging.getLogger(__name__)


def run_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    '''Удаление пустых, коротких и дубликатов по uid/text; логирование; возврат DataFrame.'''

    rows = len(df)
    df = df.copy()
    empty_mask = df['text'].isna() | (df['text'].str.strip() == '')
    n_empty = empty_mask.sum()
    df = df[~empty_mask]

    uid_dup_mask = df.duplicated(subset=['uid'])
    n_uid_dups = uid_dup_mask.sum()
    df = df[~uid_dup_mask]

    min_len = 20
    short_mask = df['text'].str.len() < min_len
    n_short = short_mask.sum()
    df = df[~short_mask]

    dup_mask = df.duplicated(subset=['text'])
    n_dups = dup_mask.sum()
    df = df[~dup_mask]
    logger.info(
        'Quality checks: rows=%d, empty=%d, duplicates(uid)=%d, short(<%d)=%d, duplicates(text)=%d, final=%d',
        rows, n_empty, n_uid_dups, min_len, n_short, n_dups, len(df),
    )
    return df.reset_index(drop=True)


MAX_LEN = 1000
CHUNK_SIZE = 600
OVERLAP = 100

def split_long_text(text: str) -> list[str]:
    '''Разбиение длинного текста на чанки (MAX_LEN, CHUNK_SIZE, OVERLAP по пробелам).'''

    if len(text) <= MAX_LEN:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        target_end = start + CHUNK_SIZE
        end = target_end
        if end < len(text):
            while end > start and text[end] != ' ':
                end -= 1
            if end == start:
                end = min(target_end, len(text))
        else:
            end = len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - OVERLAP, start + 1)
    return chunks


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    '''Strip текста и разбиение длинных строк на чанки с сохранением метаданных.'''

    df['text'] = df['text'].str.strip()
    
    rows = []
    for _, row in df.iterrows():
        for chunk in split_long_text(row['text']):
            new_row = row.copy()
            new_row['text'] = chunk
            rows.append(new_row)
    return pd.DataFrame(rows).reset_index(drop=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = load_data(URL)
    df = run_quality_checks(df)
    df = preprocess_data(df)
    df.to_parquet("data/rubq_paragraphs.parquet")
    #print(df['text'].unique())
    #print(df.head())
