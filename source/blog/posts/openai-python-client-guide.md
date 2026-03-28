---
title: "OpenAI Python SDK: Every Method with Working Examples (2026)"
description: "Tired of guessing SDK parameters? Client setup, async calls, streaming, embeddings, file uploads, and all response types — reference with examples."
date: "2026-03-13"
slug: "openai-python-client-guide"
keywords: ["OpenAI Python client", "openai Python library", "OpenAI SDK Python", "Python GPT API", "OpenAI Python guide"]
author: "Mamta Chauhan"
authorTitle: "Content Creator and AI Enthusiast"
level: "intermediate"
time: "12 min"
stack: ["Python", "OpenAI"]
---

# OpenAI Python Client Guide – Chat, Embeddings, Tools

The `openai` Python package is the standard way to interact with OpenAI's models in Python applications. It provides a clean interface for chat completions, embeddings, function calling, streaming, and batch processing. This guide covers the full range of features developers use in production.

---

## What is the OpenAI Python Client

The `openai` Python client is the official SDK for OpenAI's API. It handles authentication, HTTP requests, response parsing, streaming, and retry logic. The v1.x API (released late 2023) introduced a fully typed, resource-based interface that replaced the older function-based style.

Key features:
- Synchronous and async interfaces
- Automatic retry with exponential backoff
- Streaming support with typed event models
- Structured output via function calling
- Full type annotations compatible with mypy and pyright

---

## Why the OpenAI Python Client Matters for Developers

The Python client is the fastest path to integrating LLM capabilities into Python applications. It abstracts HTTP, auth, and error handling so you can focus on application logic. The typed interface catches configuration errors at development time rather than runtime.

For the API fundamentals and parameter reference, see [OpenAI API tutorial](/blog/openai-api-tutorial/).

---

## How the OpenAI Python Client Works

### Installation and Setup

```bash
pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],  # Default: reads OPENAI_API_KEY env var
    timeout=30.0,                          # Request timeout in seconds
    max_retries=3,                         # Auto-retry on transient errors
)
```

### Resource Structure

The client is organized into resources that mirror the API:

```python
client.chat.completions.create(...)    # Chat completions
client.embeddings.create(...)          # Embeddings
client.images.generate(...)            # Image generation
client.audio.transcriptions.create(...)# Audio transcription
client.files.create(...)               # File upload
client.fine_tuning.jobs.create(...)    # Fine-tuning
client.moderations.create(...)         # Content moderation
```

---

## Practical Examples

### Chat Completions with Type Safety

```python
from openai import OpenAI
from openai.types.chat import ChatCompletion

client = OpenAI()

def ask(prompt: str, system: str = "You are a helpful assistant.") -> str:
    response: ChatCompletion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content

# Usage tracking
result = ask("What is the capital of France?")
print(result)
```

### Structured Outputs with Pydantic

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class CodeReview(BaseModel):
    has_bugs: bool
    bugs: list[str]
    suggestions: list[str]
    overall_quality: str  # "poor" | "fair" | "good" | "excellent"

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a code reviewer."},
        {"role": "user", "content": f"Review this:\ndef div(a, b):\n    return a / b"}
    ],
    response_format=CodeReview,
)

review: CodeReview = response.choices[0].message.parsed
print(f"Has bugs: {review.has_bugs}")
print(f"Bugs: {review.bugs}")
```

### Streaming

```python
def stream_response(prompt: str):
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_response += delta

    print()  # newline at end
    return full_response

stream_response("Explain how neural networks learn in simple terms.")
```

### Async Client for Web Applications

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def ask_async(prompt: str) -> str:
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content

async def batch_questions(questions: list[str]) -> list[str]:
    """Process multiple questions concurrently."""
    tasks = [ask_async(q) for q in questions]
    return await asyncio.gather(*tasks)

# Run
results = asyncio.run(batch_questions([
    "What is RAG?",
    "What is LoRA?",
    "What is an embedding?",
]))
for q, a in zip(["RAG", "LoRA", "Embedding"], results):
    print(f"{q}: {a[:100]}...")
```

### Batch Embeddings

```python
def embed_batch(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed multiple texts in a single API call."""
    response = client.embeddings.create(model=model, input=texts)
    # Response preserves input order
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

texts = ["What is machine learning?", "How do neural networks work?", "Explain backpropagation."]
embeddings = embed_batch(texts)
print(f"Embedded {len(embeddings)} texts, each with {len(embeddings[0])} dimensions")
```

### Token Counting

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> float:
    # Prices in $ per million tokens (as of early 2026)
    prices = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }
    p = prices.get(model, prices["gpt-4o-mini"])
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

tokens = count_tokens("Explain the transformer architecture in detail.")
print(f"Tokens: {tokens}, Estimated cost: ${estimate_cost(tokens, 500):.6f}")
```

---

## Tools and Frameworks

**tiktoken** — OpenAI's tokenizer library. Count tokens before making API calls to estimate cost and avoid context overflow.

**Instructor** — Wrapper around the OpenAI client for reliable structured outputs using Pydantic. Handles validation and automatic retry on parse failures.

**LiteLLM** — Unified client that exposes an OpenAI-compatible interface for dozens of providers. Useful for model-agnostic applications.

**LangChain** — `ChatOpenAI` provides a LangChain-compatible wrapper. See [LangChain tutorial](/blog/langchain-tutorial/) for details.

For the underlying API concepts, see [OpenAI API tutorial](/blog/openai-api-tutorial/). For embeddings use in search and RAG, see [embeddings explained](/blog/embeddings-explained/).

---

## Common Mistakes

**Not using environment variables for API keys** — Always use `os.environ["OPENAI_API_KEY"]` or a `.env` file with `python-dotenv`. Never hardcode keys.

**Ignoring the response structure** — `response.choices[0].message.content` can be `None` when function calling is used. Check `tool_calls` on the message first.

**Not handling `APIError` exceptions** — Network issues, rate limits, and invalid requests all raise specific exception types. Catch them explicitly.

```python
from openai import APIError, RateLimitError, APIConnectionError

try:
    response = client.chat.completions.create(...)
except RateLimitError:
    # Wait and retry
    pass
except APIConnectionError:
    # Network issue
    pass
except APIError as e:
    print(f"API error {e.status_code}: {e.message}")
```

**Using the synchronous client in async frameworks** — In FastAPI or other async frameworks, use `AsyncOpenAI` to avoid blocking the event loop.

---

## Best Practices

- **Reuse the client instance** — Create one `OpenAI()` client at module level and reuse it. Each instantiation opens a new connection pool.
- **Use `response_format` for structured output** — More reliable than asking for JSON in plain text.
- **Set `max_tokens` explicitly** — Prevents runaway generation and controls cost.
- **Use the async client in production web apps** — Concurrent requests are significantly faster than sequential ones when using `AsyncOpenAI` with `asyncio.gather`.
- **Count tokens before large requests** — Use `tiktoken` to verify your context does not exceed the model's limit.

---

## Key Takeaways

- Create one `OpenAI()` client at module level and reuse it — each instantiation opens a new connection pool, and creating a new one per request wastes resources and adds latency.
- The v1.x SDK uses a resource-based interface organized as `client.chat.completions`, `client.embeddings`, `client.audio`, `client.images`, `client.files`, and `client.fine_tuning`.
- `response.choices[0].message.content` can be `None` when function calling is active — always check `message.tool_calls` first before reading `.content`.
- Use `AsyncOpenAI` with `asyncio.gather()` to run multiple LLM calls concurrently — sequential calls to the synchronous client in an async app block the event loop and kill throughput.
- Pydantic structured outputs via `client.beta.chat.completions.parse()` return a fully typed Python object — validation failures and parse errors are handled automatically with retries.
- Batch embeddings in a single `client.embeddings.create()` call by passing a list of strings — this is far more efficient than calling the API once per text.
- The `tiktoken` library counts tokens accurately for cost estimation and context window management before making API calls.
- Set `timeout` and `max_retries` on the client constructor to control network behavior globally rather than repeating them on every individual call.

---

## FAQ

**How do I reuse the same client across my entire application?**

Create one `client = OpenAI()` instance at module level or as a singleton. The client maintains an internal connection pool and is safe to use from multiple threads and async tasks. Avoid creating a new client per request — it adds overhead and creates new TCP connections each time.

**What is the difference between OpenAI() and AsyncOpenAI()?**

`OpenAI()` is synchronous — every API call blocks the calling thread until the response arrives. `AsyncOpenAI()` is async — every API call is a coroutine that you `await`, allowing the event loop to run other tasks while waiting for the response. Use `OpenAI()` for scripts and CLI tools; use `AsyncOpenAI()` for FastAPI, Starlette, or any other async web framework.

**How do I handle the case where the model calls a function instead of returning text?**

Check `response.choices[0].message.tool_calls` — if it is not None and not empty, the model wants to call a function. Parse `tool_call.function.arguments` as JSON to get the parameters. Execute the function, then send the result back in a new message with `role: "tool"` and the `tool_call_id`. The model will then generate a final text response using the tool result.

**Why does streaming use stream=True instead of the .stream() context manager?**

Both approaches work. `stream=True` returns a raw iterator of `ChatCompletionChunk` objects. The `.stream()` context manager (via `client.chat.completions.stream()`) provides a higher-level interface with `.text_stream` and access to the final completion via `.get_final_completion()`. The context manager is recommended for new code as it handles cleanup and provides typed access to usage statistics.

**How do I add custom headers or a proxy to the OpenAI client?**

Pass `default_headers` and `http_client` to the `OpenAI()` constructor. For a proxy, create an `httpx.Client` with your proxy settings and pass it as `http_client`. For custom headers (like organization tracking or tracing IDs), use `default_headers={"X-Custom-Header": "value"}`.

**What is the Instructor library and when should I use it?**

Instructor wraps the OpenAI client to add automatic retry on Pydantic validation failures. When the model returns JSON that does not match your schema, Instructor re-prompts with the validation error and asks the model to correct it. Use it when structured output accuracy is critical and you want automatic recovery from parse failures without writing retry logic manually.

**How do I safely shut down the async client?**

Use `AsyncOpenAI()` as an async context manager with `async with` to ensure the underlying HTTP connections are closed properly. For long-running applications, create the client once at startup and call `await client.close()` on shutdown rather than using it as a context manager per request.

---

## What to Learn Next

- [OpenAI API Tutorial](/blog/openai-api-tutorial/)
- [LLM Streaming: Real-Time Output](/blog/llm-streaming/)
- [OpenAI and LangChain: Build Pipelines Beyond Hello World](/blog/openai-langchain/)
- [Embeddings Explained](/blog/embeddings-explained/)
- [LLM API Cost Optimization](/blog/llm-api-cost-optimization/)
