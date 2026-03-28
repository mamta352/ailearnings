---
title: "RAG Chatbot: Build a Doc Q&A Bot with Full Memory (2026)"
description: "Stateless RAG breaks on follow-up questions. Build a conversational RAG chatbot with query rewriting, sliding-window memory, streaming, FastAPI backend, and Redis session store."
date: "2026-03-15"
slug: "rag-chatbot"
keywords: ["rag chatbot tutorial", "build rag chatbot python", "langchain chatbot rag", "conversational rag", "chatbot vector database", "chromadb chatbot", "rag memory chatbot", "fastapi rag chatbot"]
author: "Mamta Chauhan"
authorTitle: "Content Creator and AI Enthusiast"
updatedAt: "2026-03-28"
---

# Build a RAG Chatbot with Conversation Memory (2026)

A basic RAG pipeline answers questions. It fails the moment a user types a follow-up.

"What is the return policy?" → works fine. "Does that apply to digital products?" → retrieves chunks about digital products in general, ignores the return policy context, and returns something plausible but wrong.

This happens because standard RAG pipelines are stateless. Each question is embedded and retrieved in isolation. Follow-up questions contain pronouns and implicit references that only make sense in context — and a vector search against "does that apply to digital products?" will never find the right chunks.

The fix is not complicated: **rewrite follow-up questions into standalone questions before retrieving**, then pass both the retrieved context and the conversation history to the LLM. This guide builds a complete conversational RAG chatbot with query rewriting, sliding-window memory, streaming, a FastAPI backend, and Redis session storage for multi-user deployments.

---

## How a RAG Chatbot Differs from Q&A RAG

| Feature | Basic RAG Q&A | RAG Chatbot |
|---|---|---|
| Handles follow-ups | No | Yes — query rewriting |
| Remembers context | No | Yes — conversation memory |
| Streaming responses | Optional | Essential for UX |
| Multi-user support | Single session | Session-keyed memory |
| Query = user input | Yes | No — rewritten first |

The query rewriter is the critical addition. It turns an underspecified follow-up into a retrievable standalone question before the vector search runs.

---

## Architecture

```
User message
    ↓
Query Rewriter (LLM)
    ↓
Standalone question
    ↓
Vector retrieval (ChromaDB)
    ↓
Retrieved chunks + Conversation history
    ↓
LLM generation (streaming)
    ↓
Answer + Source citations
    ↓
Memory update (original message stored)
```

---

## Setup

```bash
pip install langchain langchain-openai langchain-community chromadb pypdf fastapi uvicorn redis
export OPENAI_API_KEY="sk-..."
```

---

## Step 1: Document Indexer

```python
# indexer.py
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

def build_index(docs_dir: str = "./docs", persist_dir: str = "./chroma_db") -> Chroma:
    loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks   = splitter.split_documents(documents)

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=persist_dir,
        collection_name="chatbot_docs"
    )
    print(f"Indexed {vs._collection.count()} chunks from {len(documents)} pages")
    return vs


def load_index(persist_dir: str = "./chroma_db") -> Chroma:
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="chatbot_docs"
    )
```

---

## Step 2: Conversation Memory

```python
# memory.py
from dataclasses import dataclass, field

@dataclass
class Message:
    role: str           # "user" or "assistant"
    content: str
    sources: list = field(default_factory=list)


class ConversationMemory:
    """Sliding-window conversation memory — keeps last N turns."""

    def __init__(self, max_turns: int = 10):
        self.messages: list[Message] = []
        self.max_turns = max_turns

    def add_user(self, content: str):
        self.messages.append(Message(role="user", content=content))

    def add_assistant(self, content: str, sources: list = None):
        self.messages.append(Message(role="assistant", content=content, sources=sources or []))

    def history_text(self, last_n: int = 6) -> str:
        """Plain text for the query rewriter."""
        recent = self.messages[-last_n:]
        return "\n".join(
            f"{'Human' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in recent
        )

    def openai_messages(self, system_prompt: str, last_n: int = 10) -> list[dict]:
        """OpenAI chat format with system prompt prepended."""
        msgs = [{"role": "system", "content": system_prompt}]
        for m in self.messages[-last_n:]:
            msgs.append({"role": m.role, "content": m.content})
        return msgs

    def clear(self):
        self.messages = []
```

---

## Step 3: Query Rewriter

The rewriter is a small, fast LLM call that runs before every retrieval. It resolves pronouns and implicit references into explicit standalone questions.

```python
# rewriter.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You rewrite follow-up questions into standalone questions. Output ONLY the rewritten question — no explanation."),
    ("human", """Conversation history:
{history}

Latest message: {question}

Rules:
- If already standalone, return it unchanged
- Resolve pronouns (it, that, them) to their referents
- Preserve the user exact intent

Standalone question:"""),
])

rewrite_chain = (
    REWRITE_PROMPT
    | ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=128)
    | StrOutputParser()
)


def rewrite_query(question: str, history: str) -> str:
    if not history.strip():
        return question
    result = rewrite_chain.invoke({"history": history, "question": question})
    return result.strip() or question
```

---

## Step 4: Core Chatbot

```python
# chatbot.py
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from memory import ConversationMemory
from rewriter import rewrite_query

SYSTEM_PROMPT = """You are a helpful assistant that answers questions from company documents.

Rules:
- Answer ONLY using the retrieved context provided in each message
- If the context does not have the answer, say: "I do not have that information in the knowledge base."
- Be concise and factual
- Do not use knowledge beyond the provided context"""


class RAGChatbot:
    def __init__(self, vectorstore: Chroma):
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        self.llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True)
        self.memory = ConversationMemory(max_turns=20)

    def _format_context(self, chunks: list) -> tuple[str, list]:
        parts, sources = [], []
        for i, c in enumerate(chunks):
            src  = c.metadata.get("source", "unknown")
            page = c.metadata.get("page", "?")
            parts.append(f"[Source {i+1}: {src}, p.{page}]\n{c.page_content}")
            sources.append({"file": src, "page": page})
        return "\n\n---\n\n".join(parts), sources

    def chat(self, user_message: str) -> dict:
        # Step 1: Rewrite to standalone question
        history_text = self.memory.history_text(last_n=6)
        standalone   = rewrite_query(user_message, history_text)

        # Step 2: Retrieve against the rewritten question
        chunks = self.retriever.invoke(standalone)
        context, sources = self._format_context(chunks)

        # Step 3: Build messages (history + retrieved context)
        messages = self.memory.openai_messages(SYSTEM_PROMPT, last_n=8)
        # Inject context with the current user message
        messages.append({
            "role": "user",
            "content": f"Context from knowledge base:\n{context}\n\nUser question: {user_message}"
        })

        # Step 4: Generate
        response = self.llm.invoke(messages)
        answer   = response.content

        # Step 5: Update memory with ORIGINAL message, not augmented version
        self.memory.add_user(user_message)
        self.memory.add_assistant(answer, sources)

        return {"answer": answer, "sources": sources, "rewritten_query": standalone}

    def stream_chat(self, user_message: str):
        """Yield string tokens, then a final metadata dict."""
        history_text = self.memory.history_text(last_n=6)
        standalone   = rewrite_query(user_message, history_text)
        chunks       = self.retriever.invoke(standalone)
        context, sources = self._format_context(chunks)

        messages = self.memory.openai_messages(SYSTEM_PROMPT, last_n=8)
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {user_message}"
        })

        full_response = []
        for chunk in self.llm.stream(messages):
            token = chunk.content
            full_response.append(token)
            yield token

        answer = "".join(full_response)
        self.memory.add_user(user_message)
        self.memory.add_assistant(answer, sources)
        yield {"__metadata__": True, "sources": sources, "rewritten_query": standalone}

    def reset(self):
        self.memory.clear()
```

---

## Step 5: CLI Interface

```python
# main.py
from pathlib import Path
from indexer import load_index, build_index
from chatbot import RAGChatbot

def run():
    vs  = load_index() if Path("./chroma_db").exists() else build_index()
    bot = RAGChatbot(vs)
    print("RAG Chatbot ready. Commands: quit | reset\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:                         continue
        if user_input.lower() in {"quit", "exit"}: break
        if user_input.lower() == "reset":
            bot.reset()
            print("History cleared.\n")
            continue

        result = bot.chat(user_input)
        print(f"\nAssistant: {result['answer']}")

        if result["sources"]:
            unique = {f"{s['file']} p.{s['page']}" for s in result["sources"]}
            print(f"Sources: {', '.join(unique)}")

        if result["rewritten_query"] != user_input:
            print(f"[Interpreted as: {result['rewritten_query']}]")
        print()

if __name__ == "__main__":
    run()
```

**Example multi-turn conversation:**

```
You: What is the return policy?
Assistant: Products can be returned within 30 days in original condition for a full refund.
Sources: handbook.pdf p.14

You: Does that apply to software licenses?
[Interpreted as: Does the return policy apply to software licenses?]
Assistant: Software licenses are non-refundable once the license key is activated (section 4.2).
Sources: handbook.pdf p.15

You: What about trial versions?
[Interpreted as: Does the return policy apply to trial versions of software licenses?]
Assistant: Trial versions are not subject to the return policy — no payment is collected during the trial.
Sources: handbook.pdf p.15
```

Without query rewriting, "Does that apply to software licenses?" would retrieve general software licensing chunks — not the return policy context.

---

## Step 6: FastAPI Backend

```python
# api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import uuid

from indexer import load_index, build_index
from chatbot import RAGChatbot

app = FastAPI(title="RAG Chatbot API")

# Load vector store once at startup
_vectorstore = load_index() if Path("./chroma_db").exists() else build_index()

# In-memory session store — replace with Redis for production
_sessions: dict[str, RAGChatbot] = {}


def get_or_create_session(session_id: str) -> RAGChatbot:
    if session_id not in _sessions:
        _sessions[session_id] = RAGChatbot(_vectorstore)
    return _sessions[session_id]


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer:          str
    sources:         list[dict]
    rewritten_query: str
    session_id:      str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    bot    = get_or_create_session(request.session_id)
    result = bot.chat(request.message)
    return ChatResponse(session_id=request.session_id, **result)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    bot = get_or_create_session(request.session_id)

    def generate():
        for item in bot.stream_chat(request.message):
            if isinstance(item, str):
                yield item

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/chat/reset")
def reset_session(session_id: str):
    if session_id in _sessions:
        _sessions[session_id].reset()
    return {"status": "ok", "session_id": session_id}


@app.get("/sessions/new")
def new_session():
    return {"session_id": str(uuid.uuid4())}


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": _vectorstore._collection.count()}
```

Run: `uvicorn api:app --reload`

---

## Step 7: Redis-Backed Session Store (Production)

In-memory sessions are lost on server restart. Use Redis for persistence and TTL-based cleanup.

```python
# redis_memory.py
import json
import redis
from memory import Message, ConversationMemory

class RedisConversationMemory(ConversationMemory):
    """Persistent memory backed by Redis with TTL-based expiry."""

    def __init__(self, session_id: str, redis_url: str = "redis://localhost:6379",
                 ttl_seconds: int = 3600, max_turns: int = 10):
        super().__init__(max_turns=max_turns)
        self.session_id  = session_id
        self.redis_key   = f"chat:session:{session_id}"
        self.ttl         = ttl_seconds
        self.r           = redis.from_url(redis_url)
        self._load()

    def _load(self):
        raw = self.r.get(self.redis_key)
        if raw:
            data = json.loads(raw)
            self.messages = [Message(**m) for m in data]

    def _save(self):
        data = [{"role": m.role, "content": m.content, "sources": m.sources}
                for m in self.messages[-self.max_turns * 2:]]
        self.r.setex(self.redis_key, self.ttl, json.dumps(data))

    def add_user(self, content: str):
        super().add_user(content)
        self._save()

    def add_assistant(self, content: str, sources: list = None):
        super().add_assistant(content, sources)
        self._save()

    def clear(self):
        super().clear()
        self.r.delete(self.redis_key)


# Use in api.py — replace in-memory sessions:
# bot = RAGChatbot(vectorstore)
# bot.memory = RedisConversationMemory(session_id, redis_url="redis://localhost:6379")
```

---

## Common Mistakes

**Storing augmented messages in memory.** If you store `"Context:\n...\n\nQuestion: ..."` in memory instead of the original user message, history balloons to thousands of tokens and the rewriter produces nonsense on subsequent turns.

**Not testing the rewriter independently.** The rewriter is a separate failure point. Log both the original and rewritten query on every turn. Rewriting errors — wrong pronoun resolution, meaning changes — are common and easy to spot if you log them.

**No session cleanup.** In-memory session dictionaries accumulate indefinitely. Implement TTL expiry in Redis or a periodic cleanup job. A chatbot with 10,000 daily users accumulates 10,000 memory objects in an in-memory dict within a day.

**Sending the full history to the rewriter.** Including all 20 turns of history in the rewriter prompt causes it to conflate topics from early in the conversation. Limit to the last 4–6 turns.

**Skipping the "I do not know" path in testing.** Test out-of-scope questions explicitly. If the chatbot answers questions that are not in your documents, the system prompt needs strengthening.

**Not bounding context window usage.** At high request volume, long conversations with many retrieved chunks can push the total prompt past the model context limit. Calculate the maximum tokens per request and set `max_tokens` and history window accordingly.

---

## Frequently Asked Questions

**How do I handle multiple concurrent users?**
Each user session needs its own `ConversationMemory` instance. In FastAPI, store sessions in a dictionary keyed by session token. For persistence across server restarts and horizontal scaling, use the `RedisConversationMemory` implementation above with TTL-based expiry.

**What happens when the query rewriter changes the meaning of a question?**
Log both the original and rewritten query in production. When you spot errors, add few-shot examples to the rewrite prompt showing the correct behavior. Few-shot prompting the rewriter is highly effective at fixing systematic rewriting mistakes.

**How many turns of history should I include?**
Six to eight turns covers the vast majority of conversational reference patterns. Beyond that, including more history increases token cost without improving quality — most users do not reference something from 15 turns ago. For the rewriter, 4–6 turns is sufficient.

**Can I use this with a local model via Ollama?**
Yes. Replace `ChatOpenAI` with `ChatOllama(model="llama3.1:8b")`. For the rewriter, a 7B model works well. For generation, use at least a 13B model for acceptable quality. The memory and rewriting logic is model-agnostic.

**Should I use LangChain ConversationalRetrievalChain instead?**
`ConversationalRetrievalChain` is deprecated in LangChain v0.2+. The custom implementation above gives you explicit control over memory management, rewriting strategy, streaming behavior, and session storage — all of which matter in production. Build your own using LCEL components.

**How do I evaluate conversational RAG quality?**
Use RAGAS on multi-turn test cases. Construct evaluation sets that include follow-up questions and check that retrieval finds the right chunks given the rewritten query. Also log `rewritten_query` vs the original in production — this is your early warning signal when the rewriter degrades.

**What is the maximum number of sessions I can run concurrently?**
With in-memory sessions: limited by available RAM — each session stores conversation history (small, typically under 50KB). With Redis-backed sessions: effectively unlimited, bounded only by Redis memory. Redis TTL handles cleanup automatically.

---

## Key Takeaways

- **Stateless RAG breaks on follow-up questions.** The fix is query rewriting — not conversation history injection alone. Rewrite before retrieving.
- **Store original messages, not augmented ones.** Storing context-injected messages in memory causes history to balloon and the rewriter to produce nonsense.
- **The rewriter is a separate failure point.** Log original vs rewritten query on every turn in production. Spot-check daily during the first week after launch.
- **Bound the history window** for both the rewriter (4–6 turns) and the LLM (8–10 turns). More history costs more tokens without improving quality.
- **Use Redis for production sessions** — in-memory session dictionaries do not survive restarts and grow unboundedly with user volume.
- **Test the "I do not know" path** with out-of-scope questions. If the chatbot answers questions not in your documents, your system prompt needs stronger grounding constraints.
- **Streaming is essential** for conversational UX — users expect to see tokens appear immediately, not wait 3–5 seconds for a full response.

---

## What to Learn Next

- **Build the basic RAG pipeline first** → [Build a RAG App: Step-by-Step](/blog/build-rag-app/)
- **Measure chatbot quality with RAGAS** → [RAG Evaluation](/blog/rag-evaluation/)
- **Scale to multiple document sources** → [Multi-Document RAG](/blog/multi-document-rag/)
- **Run the LLM locally for privacy** → [Ollama Guide](/ollama-local-llm-guide/)
- **Deploy to production** → [Production RAG Patterns](/blog/production-rag/)
