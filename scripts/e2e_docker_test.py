import sys
import time

import requests


def wait_ready(
    base_url: str = 'http://localhost:8000',
    ready_path: str = '/ready',
    timeout: int = 120,
    poll_interval: float = 5.0,
) -> bool:
    '''Ждет, пока сервис вернёт 200 на GET /ready (модель загружена).'''
    url = base_url.rstrip('/') + ready_path
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(poll_interval)
    return False


def run_e2e_query_test(
    url: str = 'http://localhost:8000/query',
    question: str = 'Кто такой Юрий Гагарин?',
    timeout: int = 120,
    wait_for_ready: bool = True,
    ready_timeout: int = 120,
) -> int:
    '''E2E-тест: ожидание /ready, POST /query, проверка ответа; возвращает 0 при успехе, 1 при ошибке.'''
    base_url = url.replace('/query', '')
    if wait_for_ready:
        print('Waiting for query service to be ready (/ready)')
        if not wait_ready(base_url=base_url, timeout=ready_timeout):
            print('Timeout: /ready did not return 200')
            return 1
        print('Service is ready.')
    payload = {
        "question": question,
        "top_k": 3,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        print(f'Request to {url} failed: {exc}')
        return 1

    if resp.status_code != 200:
        print(f'Unexpected status code: {resp.status_code}, body={resp.text}')
        return 1

    data = resp.json()

    answer = data.get('answer', '')
    chunks = data.get('chunks', [])

    if not isinstance(answer, str) or not answer.strip():
        print('Response "answer" is empty or missing')
        return 1

    if not isinstance(chunks, list) or not chunks:
        print('Response "chunks" is empty or missing')
        return 1

    print('E2E Docker test passed.')
    print(f'Question: {data.get('question')}')
    print(f'Answer (truncated): {answer[:200]!r}')
    print(f'Chunks returned: {len(chunks)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(run_e2e_query_test())

