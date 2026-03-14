import logging
from typing import Any, Dict

import requests

from src.config.config import HF_API_TOKEN, HF_LLM_MAX_NEW_TOKENS, HF_LLM_MODEL


logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    '''Ошибка при обращении к LLM.'''


def generate_answer(
    prompt: str,
    max_new_tokens: int | None = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    timeout: int = 60,
) -> str:
    '''Запрашивает ответ у LLM через Hugging Face Router (Chat Completions API).'''

    if not HF_API_TOKEN:
        raise LLMError('HF_API_TOKEN is not set')
    if max_new_tokens is None:
        max_new_tokens = HF_LLM_MAX_NEW_TOKENS

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    url = 'https://router.huggingface.co/v1/chat/completions'
    payload: Dict[str, Any] = {
        "model": HF_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise LLMError('Failed to call Hugging Face Inference API') from exc

    if resp.status_code == 503:
        raise LLMError('LLM service temporarily unavailable (model loading)')

    if not resp.ok:
        raise LLMError(f'HF Inference API error: {resp.status_code}')

    try:
        data = resp.json()
    except ValueError as exc:
        logger.error('Failed to parse HF response JSON: %s', resp.text[:500])
        raise LLMError('Failed to parse LLM response') from exc

    choices = data.get('choices')
    if isinstance(choices, list) and choices:
        msg = choices[0].get('message')
        if isinstance(msg, dict):
            content = msg.get('content')
            if isinstance(content, str):
                return content.strip()

    raise LLMError('Unexpected LLM response format')

