import pandas as pd

URL = 'https://raw.githubusercontent.com/vladislavneon/RuBQ/refs/heads/master/RuBQ_2.0/RuBQ_2.0_paragraphs.json'


def load_data(url: str) -> pd.DataFrame:
    '''Загрузка JSON по URL в DataFrame.'''
    return pd.read_json(url)