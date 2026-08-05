"""
LLM client — Groq API with retry logic, error handling, streaming support,
response caching, and structured tracing.
"""
import json
import logging
import time
from typing import Generator, List, Optional

import httpx

from src.config import GROQ_API_KEY, GROQ_MODEL, GROQ_URL
from src.cache import get_cached_response, set_cached_response
from src.tracing import trace_span

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5


def _build_messages(system: str, history: List[dict], user_message: str) -> List[dict]:
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


def call_llm(
    prompt: str,
    history: Optional[List[dict]] = None,
    system: str = "You are a helpful medical AI assistant. Answer clearly and in plain English.",
    temperature: float = 0.3,
    response_format: Optional[dict] = None,
    use_cache: bool = True,
) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")

    # Caching only applies to single-turn, cacheable calls (no conversation
    # history) — a follow-up question with prior context is not a repeat.
    cacheable = use_cache and not history
    if cacheable:
        cached = get_cached_response(system, prompt, GROQ_MODEL, temperature)
        if cached is not None:
            with trace_span("llm_call", model=GROQ_MODEL, cache_hit=True,
                             prompt_chars=len(prompt), system_chars=len(system)):
                pass
            return cached

    messages = _build_messages(system, history or [], prompt)
    last_error = None

    body = {"model": GROQ_MODEL, "messages": messages, "temperature": temperature}
    if response_format:
        body["response_format"] = response_format

    with trace_span("llm_call", model=GROQ_MODEL, cache_hit=False,
                     prompt_chars=len(prompt), system_chars=len(system)) as span:
        for attempt in range(MAX_RETRIES + 1):
            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.post(
                        GROQ_URL,
                        headers={
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                    )
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    span["output_chars"] = len(content)
                    span["attempts"] = attempt + 1
                    if cacheable:
                        set_cached_response(system, prompt, GROQ_MODEL, temperature, content)
                    return content

            except httpx.HTTPStatusError as e:
                if e.response.status_code in (400, 401, 403):
                    raise RuntimeError(f"LLM API error {e.response.status_code}: {e.response.text[:200]}")
                last_error = e
                logger.warning("Groq transient error (attempt %d): %s", attempt + 1, e)

            except httpx.RequestError as e:
                last_error = e
                logger.warning("Groq network error (attempt %d): %s", attempt + 1, e)

            except (KeyError, IndexError):
                raise RuntimeError("Unexpected response from LLM API.")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

        raise RuntimeError(f"LLM request failed after {MAX_RETRIES + 1} attempts: {last_error}")


def stream_llm(
    prompt: str,
    history: Optional[List[dict]] = None,
    system: str = "You are a helpful medical AI assistant. Answer clearly and in plain English.",
) -> Generator[str, None, None]:
    if not GROQ_API_KEY:
        yield "[Error: GROQ_API_KEY not set]"
        return

    messages = _build_messages(system, history or [], prompt)

    for attempt in range(MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=60) as client:
                with client.stream(
                    "POST", GROQ_URL,
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"model": GROQ_MODEL, "messages": messages, "stream": True, "temperature": 0.3},
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            payload = line[6:]
                            if payload == "[DONE]":
                                return
                            try:
                                chunk = json.loads(payload)
                                delta = chunk["choices"][0]["delta"].get("content", "")
                                if delta:
                                    yield delta
                            except (json.JSONDecodeError, KeyError):
                                continue
                    return

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 401, 403):
                yield f"\n[Error: {e.response.status_code}]"
                return
            logger.warning("Stream error (attempt %d): %s", attempt + 1, e)

        except httpx.RequestError as e:
            logger.warning("Network error (attempt %d): %s", attempt + 1, e)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
        else:
            yield "\n[Stream failed after retries — please try again]"
