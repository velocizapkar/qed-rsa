"""Async batched generation via OpenAI-compatible client talking to SGLang."""

import asyncio
from typing import List

from openai import AsyncOpenAI


def get_client(port: int = 30000) -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointing at the local SGLang server."""
    return AsyncOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="unused",
    )


async def generate_one(
    client: AsyncOpenAI,
    messages: List[dict],
    model: str = "default",
    max_tokens: int = 16384,
    temperature: float = 0.7,
) -> str:
    """Generate a single completion."""
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


async def generate_batch(
    client: AsyncOpenAI,
    messages_list: List[List[dict]],
    model: str = "default",
    max_tokens: int = 16384,
    temperature: float = 0.7,
    max_concurrent: int = 32,
) -> List[str]:
    """Generate completions for a batch of message lists concurrently.

    Uses a semaphore to limit concurrency and avoid overwhelming the server.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _gen(msgs: List[dict]) -> str:
        async with sem:
            return await generate_one(
                client, msgs, model=model,
                max_tokens=max_tokens, temperature=temperature,
            )

    tasks = [_gen(msgs) for msgs in messages_list]
    return await asyncio.gather(*tasks)


def generate_batch_sync(
    messages_list: List[List[dict]],
    port: int = 30000,
    model: str = "default",
    max_tokens: int = 16384,
    temperature: float = 0.7,
    max_concurrent: int = 32,
) -> List[str]:
    """Synchronous wrapper around generate_batch."""
    client = get_client(port=port)
    return asyncio.run(
        generate_batch(
            client, messages_list,
            model=model, max_tokens=max_tokens,
            temperature=temperature, max_concurrent=max_concurrent,
        )
    )
