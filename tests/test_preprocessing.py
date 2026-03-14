import pandas as pd

from src.data.preprocessing import (
    run_quality_checks,
    split_long_text,
    preprocess_data,
    MAX_LEN,
)


def test_run_quality_checks_filters_empty_short_and_duplicates():
    '''Проверяет, что run_quality_checks удаляет пустые, короткие и дубликаты по тексту.'''
    df = pd.DataFrame(
        {
            "uid": [1, 2, 3, 4],
            "ru_wiki_pageid": [10, 10, 11, 11],
            "text": [
                "   ",
                "short",
                "valid text long enough",
                "valid text long enough",
            ],
        }
    )

    cleaned = run_quality_checks(df)

    assert len(cleaned) == 1
    assert cleaned["text"].iloc[0] == "valid text long enough"


def test_split_long_text_respects_max_len_and_overlap():
    '''Проверяет, что split_long_text даёт чанки не длиннее MAX_LEN и несколько чанков при длинном тексте.'''
    base = "abcd " * 300
    assert len(base) > MAX_LEN

    chunks = split_long_text(base)

    assert all(len(c) <= MAX_LEN for c in chunks)
    assert len(chunks) > 1


def test_preprocess_data_preserves_metadata_and_splits_long_rows():
    '''Проверяет, что preprocess_data разбивает длинные строки на чанки и сохраняет uid/pageid.'''
    long_text = "x" * (MAX_LEN + 100)
    df = pd.DataFrame(
        {
            "uid": [1],
            "ru_wiki_pageid": [42],
            "text": [long_text],
        }
    )

    processed = preprocess_data(df)

    assert len(processed) > 1
    assert processed["uid"].nunique() == 1
    assert processed["ru_wiki_pageid"].nunique() == 1

