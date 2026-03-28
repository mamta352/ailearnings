---
title: "Build a RAG App: From PDF to Answer with Full Code (2026)"
description: "RAG tutorials without working code waste time. This one ships — ingest PDFs, embed with OpenAI, store in ChromaDB, query with LCEL. Full pipeline, FastAPI backend, and retrieval tuning."
date: "2026-03-13"
slug: "build-rag-app"
keywords: ["build RAG app", "RAG application tutorial", "retrieval augmented generation tutorial", "LangChain RAG", "vector database app", "LCEL RAG pipeline"]
author: "Mamta Chauhan"
authorTitle: "Content Creator and AI Enthusiast"
updatedAt: "2026-03-28"
---

# How to Build a RAG Application Step-by-Step (2026)

You gave GPT-4o your 200-page policy document. It hallucinated three paragraphs that contradict page 47. You tried a longer system prompt. Still wrong.

The problem is not the model. The problem is that language models cannot reliably answer questions from documents they have not seen. Fine-tuning embeds knowledge into weights — it cannot retrieve specific passages or cite sources. **Retrieval-Augmented Generation (RAG) solves this** by fetching the relevant passages at query time and feeding them to the model as context.

This guide builds a complete, working RAG application from scratch: document ingestion, vector store, LCEL retrieval chain, FastAPI backend, and retrieval tuning. All code is copy-paste ready.

---

## What You Are Building

A document Q&A assistant that:
- Loads and indexes PDF documents into a vector store
- Accepts natural language questions
- Retrieves the most relevant document passages
- Generates grounded answers that cite sources
- Returns "I do not have that information" when the answer is not in the documents

This pattern generalizes to customer support bots, internal knowledge bases, documentation assistants, legal research tools, and code search.

---

## How RAG Works

RAG has two clearly separated phases:

**Indexing phase (offline — run once):**
1. Load documents (PDF, web pages, text files)
2. Split into overlapping chunks (400–600 characters each)
3. Embed each chunk using an embedding model → vector
4. Store vectors in a vector database

**Query phase (online — per user request):**
1. Embed the user's question → vector
2. Search the vector database for the most similar chunk vectors (cosine similarity)
3. Insert retrieved chunks into the LLM prompt as context
4. Generate an answer grounded in that context

The retrieval step is what separates RAG from standard chat: the model only sees what you retrieve, not its full training-data knowledge about the topic.

---

## Prerequisites and Cost Estimate

```bash
pip install langchain langchain-openai langchain-community \
            chromadb pypdf fastapi uvicorn
export OPENAI_API_KEY="sk-..."
```

**Cost to index 500 chunks (~100 pages of PDF):**

| Operation | Model | Cost |
|---|---|---|
| Embedding 500 chunks | text-embedding-3-small | ~$0.002 |
| Query (per question) | gpt-4o-mini + 4 chunks context | ~$0.001 |
| Query (per question) | gpt-4o + 4 chunks context | ~$0.01 |

Index once. Cost is negligible. The query cost is what adds up at scale.

---

## Step 1: Load and Chunk Documents

Document loading converts file formats into LangChain `Document` objects. Chunking breaks them into segments small enough for an embedding model to represent meaningfully.

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load a single PDF
loader = PyPDFLoader("docs/employee-handbook.pdf")
documents = loader.load()

# Or load all PDFs from a directory
# loader = DirectoryLoader("docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
# documents = loader.load()

print(f"Loaded {len(documents)} pages")
print(f"First page preview: {documents[0].page_content[:300]}")
print(f"Metadata: {documents[0].metadata}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,           # overlap prevents information loss at boundaries
    separators=["\n\n", "\n", " ", ""]  # prefer paragraph breaks
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks from {len(documents)} pages")
```

**Chunk size is the most impactful parameter in RAG.** Small chunks (256–512 chars) produce precise retrieval but lose surrounding context. Large chunks (1024–2048 chars) preserve context but dilute the embedding signal — a 2,000-char chunk contains many ideas and matches many queries weakly rather than one query strongly. Start at 400–500 characters, then tune based on your retrieval quality metrics.

Other loader options: `WebBaseLoader` (URLs), `TextLoader` (.txt), `UnstructuredFileLoader` (Word, CSV, HTML).

---

## Step 2: Create the Vector Store

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Index all chunks — calls the embeddings API once per chunk
# Run this once; the result persists to disk
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_documents"
)

print(f"Indexed {vectorstore._collection.count()} chunks")
print("Vector store persisted to ./chroma_db")
```

Embedding 500 chunks with `text-embedding-3-small` costs roughly $0.002 and takes about 10 seconds. Once persisted, never re-index unless documents change.

**To reload the persisted store on subsequent runs:**

```python
# In production: load from disk — never re-index on every startup
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="my_documents"
)
print(f"Loaded {vectorstore._collection.count()} indexed chunks")
```

---

## Step 3: Build the Retrieval Chain with LCEL

LangChain's modern approach uses LCEL (LangChain Expression Language) — the `|` pipe syntax. The old `RetrievalQA.from_chain_type()` is legacy and should not be used in new code.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# The grounding prompt is the most critical piece.
# Without explicit constraints, the model uses training-data knowledge.
SYSTEM_PROMPT = """You are a document assistant. Answer based ONLY on the context below.
If the context does not contain enough information to answer, say exactly:
"I do not have enough information to answer that."

Do not invent or infer information not present in the context.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}   # retrieve top 4 most relevant chunks
)

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', '?')}\n{doc.page_content}"
        for doc in docs
    )

# LCEL chain — readable, composable, streamable
rag_chain = (
    {
        "context":  retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

**Why LCEL over RetrievalQA?**

| Feature | RetrievalQA (legacy) | LCEL |
|---|---|---|
| Streaming | No | Yes — `rag_chain.stream(question)` |
| Async | Limited | Yes — `rag_chain.ainvoke(question)` |
| Batch | No | Yes — `rag_chain.batch([q1, q2])` |
| Source docs | Awkward | Handle explicitly with full control |
| Tracing | Partial | Full LangSmith support |

---

## Step 4: Query the Application

```python
def ask(question: str) -> dict:
    """Ask a question and return the answer with source citations."""
    # Retrieve source docs separately for citation
    source_docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)

    return {
        "answer": answer,
        "sources": [
            {
                "page":    doc.metadata.get("page", "?"),
                "source":  doc.metadata.get("source", "unknown"),
                "snippet": doc.page_content[:200] + "..."
            }
            for doc in source_docs
        ]
    }

# Test it
response = ask("What is the company vacation policy?")
print(response["answer"])
print("\nSources:")
for src in response["sources"]:
    print(f"  {src['source']}, page {src['page']}")
    print(f"  \"{src['snippet']}\"")

# Test the "I do not know" path — this is as important as happy-path tests
response = ask("What is the square root of 144?")
print(response["answer"])
# Should say: "I do not have enough information to answer that."
```

**Streaming responses** (better UX for long answers):

```python
# Stream token by token — no waiting for the full response
for token in rag_chain.stream("What is the vacation policy?"):
    print(token, end="", flush=True)
print()
```

---

## Step 5: Add a FastAPI Backend

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Document Q&A API")

# Load vector store once at startup — not on every request
_vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="my_documents"
)
_retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})
_rag_chain  = (
    {"context": _retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    source_docs = _retriever.invoke(request.question)
    answer      = _rag_chain.invoke(request.question)

    return {
        "answer": answer,
        "sources": [
            {"source": doc.metadata.get("source"), "page": doc.metadata.get("page")}
            for doc in source_docs
        ]
    }


@app.post("/ask/stream")
def ask_stream(request: QuestionRequest):
    """Stream the answer token by token."""
    def generate():
        for token in _rag_chain.stream(request.question):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": _vectorstore._collection.count()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Improving Retrieval Quality

The default similarity search works. These techniques improve precision and recall for production.

### MMR (Maximal Marginal Relevance)

Standard top-K retrieval often returns near-duplicate chunks (adjacent paragraphs from the same section). MMR reranks results to maximize diversity while maintaining relevance.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,             # return 5 diverse results
        "fetch_k": 20,      # from 20 initial candidates
        "lambda_mult": 0.5  # balance: relevance (1.0) vs diversity (0.0)
    }
)
```

### Metadata Filtering

Filter by document source when the query implies a specific document.

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {"source": "employee-handbook.pdf"}
    }
)
```

### Similarity Score Threshold

Reject chunks that are only weakly related. This prevents the model from generating answers from tangentially related content.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 4}
)
```

### Hybrid Search (Keyword + Semantic)

Pure semantic search misses exact-match queries (product codes, names, technical terms). Hybrid search combines BM25 keyword search with vector similarity for much better recall on mixed query types.

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# BM25 for keyword matching
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

# Vector store for semantic matching
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Combine: 40% keyword weight, 60% semantic weight
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)
```

Use hybrid search when your documents contain proper nouns, product names, or technical codes that need exact matching.

---

## Common Mistakes

**Re-indexing on every startup** — Embedding 500 documents takes 30 seconds and costs money. Index once, persist to disk, and load the persisted store at startup. This is the most common performance mistake in RAG applications.

**Chunk size too large** — A 2,000-character chunk contains many ideas. The embedding averages over all of them, making the chunk match many queries weakly rather than one strongly. Start at 400–500 characters.

**No chunk overlap** — Information at a chunk boundary is split in half. With `chunk_overlap=0`, a sentence that spans the boundary is lost. Use 10–15% overlap relative to chunk size.

**Weak grounding prompt** — Without explicit constraints, GPT-4o supplements retrieved passages with its training data, producing confident answers not in your documents. The system prompt must say: "answer ONLY based on the context."

**No source attribution** — Always return source document metadata with the answer. Users need to verify answers, and citations dramatically increase trust.

**Only testing happy-path queries** — Test with out-of-scope questions to verify the model correctly says "I do not know." If it does not, strengthen the grounding constraint.

**Using the same embedding model for different use cases without benchmarking** — `text-embedding-3-small` is cheapest but `text-embedding-3-large` has better recall for technical content. Run RAGAS evaluations on both before committing.

---

## Performance Benchmarks

| Embedding Model | Dimensions | MTEB Score | Cost/1M tokens |
|---|---|---|---|
| text-embedding-3-small | 1536 | 62.3 | $0.02 |
| text-embedding-3-large | 3072 | 64.6 | $0.13 |
| nomic-embed-text (local) | 768 | 62.0 | Free |

For most RAG applications, `text-embedding-3-small` provides the best cost-to-quality ratio. Use `text-embedding-3-large` only when retrieval quality is measurably worse.

---

## Frequently Asked Questions

**What is the difference between RAG and fine-tuning?**
RAG retrieves information at query time from an external store and includes it in the LLM's context. Fine-tuning bakes knowledge into model weights during training. RAG is better for: private/dynamic data, citation requirements, and frequent updates. Fine-tuning is better for: consistent output style, domain-specific reasoning patterns, and when you have thousands of labeled examples. Most production systems use RAG first, and fine-tune only if RAG does not meet quality requirements.

**How many chunks should I retrieve (k value)?**
Start with k=4. More chunks give the model more context but also more irrelevant noise. If your answers are too generic, lower k. If the model says "I do not have information" on questions your documents cover, raise k to 6–8. Always measure with RAGAS metrics rather than guessing.

**Can I use a local embedding model instead of OpenAI?**
Yes. Replace `OpenAIEmbeddings` with `OllamaEmbeddings(model="nomic-embed-text")` for a fully free, local pipeline. `nomic-embed-text` has MTEB scores comparable to `text-embedding-3-small`. This requires Ollama running locally — see our [Ollama guide](/ollama-local-llm-guide/).

**Why does my RAG app answer questions that are not in my documents?**
Your grounding prompt is too weak. Add "Do not use any knowledge outside the provided context" and "If the answer is not in the context, say: I do not have enough information." Test with 10–20 out-of-scope questions to validate.

**How do I update documents without re-indexing everything?**
Use ChromaDB's `add_documents()` to append new chunks, and `delete()` with a metadata filter to remove outdated ones. Design your metadata schema upfront to include a document ID or version field so you can selectively update.

**How much does it cost to run a RAG app for 1,000 users per day?**
With `text-embedding-3-small` (queries only, no re-indexing) and `gpt-4o-mini`, each query costs roughly $0.001–$0.003. At 1,000 queries per day: ~$1–3/day or $30–90/month. Switch to a local Ollama model for embeddings and the only cost is the LLM query.

**What vector database should I use in production?**
ChromaDB (local/free) for development and small deployments under 100K chunks. Pinecone (managed, free tier) for medium scale. Qdrant (self-hosted or cloud) for large scale with filtering requirements. Weaviate for multi-modal data. The choice matters less than your indexing strategy and retrieval quality.

---

## Key Takeaways

- RAG has two phases: **indexing** (offline, run once) and **querying** (online, per request). Never re-index on startup.
- **Chunk size (400–500 chars)** and **chunk overlap (10–15%)** are the most impactful parameters. Tune these first before anything else.
- Use **LCEL syntax** (`prompt | llm | StrOutputParser()`) — `RetrievalQA.from_chain_type()` is deprecated in LangChain v0.2+.
- The **grounding prompt** is what keeps the model from hallucinating. It must explicitly restrict answers to the provided context.
- **Always return source citations.** Without them, users cannot verify answers and trust collapses.
- **Test the "I do not know" path.** If the model answers out-of-scope questions, your grounding prompt needs work.
- **Hybrid search** (BM25 + vector) outperforms pure semantic search on documents with proper nouns, codes, and technical terms.

---

## What to Learn Next

- **Measure whether your RAG actually works** → [RAG Evaluation with RAGAS](/blog/rag-evaluation/)
- **Understand why RAG works** → [RAG Explained](/blog/rag-explained/)
- **Go deeper on vector databases** → [Vector Database Guide](/blog/vector-database-guide/)
- **Advanced chunking strategies** → [Document Chunking Strategies](/blog/document-chunking-strategies/)
- **LangChain for complex retrieval pipelines** → [LangChain Tutorial](/blog/langchain-tutorial/)
- **Multi-document RAG with citations** → [Multi-Document RAG](/blog/multi-document-rag/)
