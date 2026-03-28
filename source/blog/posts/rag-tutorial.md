---
title: "RAG Tutorial Hub: Everything to Build Retrieval Pipelines (2026)"
description: "Do not piece together RAG from 10 different tutorials. Architecture, chunking, embedding, evaluation, and production."
date: "2026-03-15"
slug: "rag-tutorial"
keywords: ["rag tutorial", "retrieval augmented generation tutorial", "langchain rag", "build rag system", "rag python", "chromadb tutorial", "openai rag"]
author: "Mamta Chauhan"
authorTitle: "Content Creator and AI Enthusiast"
updatedAt: "2026-03-15"
---

# RAG Tutorial: Build Retrieval-Augmented Generation Systems (2026)

The gap between a GPT-4o demo and a production-grade document assistant is mostly in retrieval. Developers who try to skip the fundamentals and go straight to "connect my PDFs to ChatGPT" end up with a system that works great on their test documents and fails on everything else. The failure mode is usually retrieval — wrong chunks, too many chunks, or chunks that match semantically but miss the user's actual intent.

This tutorial walks through building a complete RAG system from scratch — document ingestion, chunking, embedding, vector search, and grounded generation. Every code block is runnable. The goal is not just a working demo but a system where you understand why each piece exists and what to do when it breaks.

Before diving in, if you want to understand the architecture decisions behind these choices, read the [RAG Architecture Guide](/blog/rag-architecture-guide) first.

---

## Concept Overview

**Retrieval-Augmented Generation** augments an LLM's context with content retrieved from a search index at query time. The LLM generates answers grounded in retrieved content rather than relying on memorized training-data knowledge.

The system has two independently deployable pieces:
- **Indexer** — processes documents, creates embeddings, populates the vector store
- **Query engine** — takes user questions, retrieves relevant chunks, calls the LLM

This separation matters. The indexer runs offline (as a batch job or triggered by document uploads). The query engine runs online, per request. Mixing them creates operational headaches.

---

## How It Works

![Architecture diagram](/assets/diagrams/rag-tutorial-diagram-1.png)

The embedding model is the bridge between the two phases. It must be identical in both — index with `text-embedding-3-small`, query with `text-embedding-3-small`. Switching models after indexing silently breaks retrieval because the embedding spaces are incompatible.

---

## Prerequisites

```bash
pip install langchain langchain-openai langchain-community \
            chromadb pypdf sentence-transformers

export OPENAI_API_KEY="sk-..."
```

Create a `docs/` directory and put a few PDF files in it. Any PDFs work — product documentation, research papers, company policies.

---

## Implementation Example

### Step 1: Document Loading

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from pathlib import Path

def load_documents(docs_dir: str) -> list:
    """Load all PDFs from a directory into LangChain Document objects."""
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True   # parallel loading for large collections
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {docs_dir}")
    return documents

documents = load_documents("./docs")

# Inspect what we loaded
print(documents[0].metadata)   # {'source': './docs/handbook.pdf', 'page': 0}
print(documents[0].page_content[:300])
```

LangChain's `DirectoryLoader` handles mixed document types if you configure multiple loaders. In practice, most teams start with PDFs and expand to web pages, Notion exports, and database records over time.

### Step 2: Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents: list, chunk_size: int = 512, overlap: int = 64) -> list:
    """
    Split documents into overlapping chunks.

    chunk_size: target character count per chunk
    overlap: characters shared between adjacent chunks (prevents boundary loss)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (avg {sum(len(c.page_content) for c in chunks) // len(chunks)} chars)")
    return chunks

chunks = chunk_documents(documents)

# Check chunk distribution
sizes = [len(c.page_content) for c in chunks]
print(f"Min: {min(sizes)}, Max: {max(sizes)}, Avg: {sum(sizes)//len(sizes)}")
```

One thing many developers overlook: check your chunk size distribution before indexing. If you have many chunks under 100 characters, your splitter is fragmenting sentences. If most chunks are at the max size, the separators aren't finding natural break points — consider adding domain-specific separators.

### Step 3: Embedding and Indexing

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import time

def build_vector_store(chunks: list, persist_dir: str) -> Chroma:
    """Embed chunks and persist to ChromaDB."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        # Batch size controls API call grouping — 1000 is the OpenAI max
        chunk_size=1000
    )

    start = time.time()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="documents",
        collection_metadata={"hnsw:space": "cosine"}   # cosine similarity
    )
    elapsed = time.time() - start

    count = vectorstore._collection.count()
    print(f"Indexed {count} vectors in {elapsed:.1f}s")
    print(f"Estimated cost: ${count * 0.00002:.4f}")   # $0.02/1M tokens approx
    return vectorstore

vs = build_vector_store(chunks, "./chroma_db")
```

### Step 4: Retriever Configuration

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def load_vector_store(persist_dir: str) -> Chroma:
    """Load a persisted vector store — use this in production, not from_documents()."""
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="documents"
    )

vs = load_vector_store("./chroma_db")

# Configure the retriever
retriever = vs.as_retriever(
    search_type="mmr",          # Maximal Marginal Relevance for diversity
    search_kwargs={
        "k": 5,                 # final number of chunks to return
        "fetch_k": 20,          # candidate pool size before MMR reranking
        "lambda_mult": 0.6      # 1.0 = pure similarity, 0.0 = pure diversity
    }
)

# Test retrieval directly before wiring up the LLM
test_chunks = retriever.invoke("What is the refund policy?")
for i, chunk in enumerate(test_chunks):
    print(f"\nChunk {i+1} [{chunk.metadata.get('source', '?')}, p{chunk.metadata.get('page', '?')}]")
    print(chunk.page_content[:200])
```

Always test retrieval before testing end-to-end generation. If the retrieved chunks are wrong, no prompt engineering will save you. This is where most RAG debugging time goes.

### Step 5: QA Chain with Grounding Prompt

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

GROUNDING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant answering questions from company documents.

RULES:
- Answer ONLY using the context provided below
- If the context does not contain the answer, say: "I do not have that information in my knowledge base."
- Be concise and factual
- Do not reference or use any knowledge outside the context

Context:
{context}"""),
    ("human", "{question}"),
])

def build_rag_chain(retriever):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,       # deterministic answers for factual Q&A
        max_tokens=512       # prevent runaway generation
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | GROUNDING_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

rag_chain = build_rag_chain(retriever)
```

### Step 6: Query and Response Formatting

```python
def ask(rag_chain, retriever, question: str, verbose: bool = False) -> dict:
    """
    Query the RAG chain and return a structured response.

    Returns:
        dict with 'answer', 'sources', and 'chunks_used'
    """
    retrieved_docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)

    sources = []
    seen = set()
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}:{page}"
        if key not in seen:
            seen.add(key)
            sources.append({"file": source, "page": page})

    if verbose:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Sources: {sources}")
        print(f"Chunks retrieved: {len(retrieved_docs)}")

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(retrieved_docs)
    }


# Run test queries
test_questions = [
    "What is the return policy for electronics?",      # in-scope
    "How many vacation days do employees get?",         # in-scope
    "What is the GDP of Germany?",                      # out-of-scope
    "Who is the CEO of the company?",                   # may or may not be in docs
]

for q in test_questions:
    ask(rag_chain, retriever, q, verbose=True)
```

### Complete Pipeline Script

```python
# complete_rag_pipeline.py
import os
from pathlib import Path

def main():
    DOCS_DIR = "./docs"
    PERSIST_DIR = "./chroma_db"

    # Decide whether to re-index or load existing
    if not Path(PERSIST_DIR).exists():
        print("Building index from scratch...")
        docs = load_documents(DOCS_DIR)
        chunks = chunk_documents(docs)
        vs = build_vector_store(chunks, PERSIST_DIR)
    else:
        print("Loading existing index...")
        vs = load_vector_store(PERSIST_DIR)
        print(f"Loaded {vs._collection.count()} vectors")

    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    rag_chain = build_rag_chain(retriever)

    # Interactive REPL
    print("\nRAG System ready. Type 'quit' to exit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        result = ask(rag_chain, retriever, question, verbose=True)

if __name__ == "__main__":
    main()
```

---

## Best Practices

**Start with fewer, better documents.** A focused knowledge base of 50 high-quality documents produces better retrieval than 500 documents of mixed quality. Clean your source documents before indexing.

**Validate retrieval separately from generation.** Write a set of test questions with known answers and measure retrieval precision — are the right chunks appearing in the top-5 results? Fix retrieval failures before optimizing the prompt.

**Use metadata aggressively.** Store document name, section, date, author, and any other fields in chunk metadata. You can filter by these during retrieval (`search_kwargs={"filter": {"source": "handbook.pdf"}}`) to scope queries to the right documents.

**Keep chunk sizes consistent.** Mixed chunk sizes produce uneven embedding quality. If you're loading multiple document types, configure type-specific splitters and verify chunk size distributions independently.

**Cache embeddings for repeated documents.** OpenAI's embedding API is cheap but not free. Use `CacheBackedEmbeddings` from LangChain to avoid re-embedding unchanged documents.

```python
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

store = LocalFileStore("./embedding_cache")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    OpenAIEmbeddings(model="text-embedding-3-small"),
    store,
    namespace="text-embedding-3-small"
)
```

---

## Common Mistakes

**Embedding the question with a different model than the index.** The similarity score between a query embedding and a chunk embedding is only meaningful when both use the same model. This is the most common silent failure in RAG systems.

**Not testing the "I do not know" path.** If out-of-scope questions produce hallucinated answers, the grounding prompt needs to be stronger. Add explicit examples of when to decline.

**Running indexing on every API request.** Indexing is slow and expensive. The vector store should be built once and persisted. Reload from disk at startup.

**Using `k=10` by default "to be safe."** More chunks increases recall but hurts precision and fills your context window with noise. Start at 4–5 and measure.

**Not handling empty retrieval.** If the vector store returns zero chunks (below threshold), gracefully decline instead of sending an empty context to the LLM.

---

## Frequently Asked Questions

**What is the minimum viable RAG setup?**
A single Python script with LangChain, ChromaDB, and OpenAI handles up to ~50,000 chunks on a developer laptop. For production, separate the indexing job from the query API, use Pinecone or Qdrant for the vector store, and add Redis for query caching.

**Can I use a local LLM instead of OpenAI?**
Yes. Replace `ChatOpenAI` with `ChatOllama` from `langchain-ollama` and point it at a locally running Ollama instance. For embeddings, use `HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")`. The LCEL chain structure is identical — just swap the LLM and embedding objects.

**How often should I re-index?**
Re-index when documents change. For frequently changing content, build an incremental pipeline that detects new or modified files via content hashing and updates only those chunks. Full re-indexing nightly works for most knowledge bases under 100,000 documents.

**What is the difference between MMR and similarity search?**
Standard similarity search returns the K chunks with the highest cosine similarity to the query — which often means 5 chunks from the same paragraph. MMR (Maximal Marginal Relevance) balances relevance with diversity, selecting chunks that are both relevant to the query and different from each other. Use MMR by default to avoid redundant context.

**How do I handle multiple languages?**
Use a multilingual embedding model like `BAAI/bge-m3` or `multilingual-e5-large`. Store the document language in metadata. Embed the query in its original language — multilingual models handle cross-lingual matching reasonably well.

**Why should I test out-of-scope questions?**
In production, users will ask questions your knowledge base cannot answer. Without explicit "I do not know" testing, you may not discover that your grounding prompt is too weak until real users start getting hallucinated answers. Test at least 10 out-of-scope queries before deploying.

**What is `CacheBackedEmbeddings` and do I need it?**
`CacheBackedEmbeddings` stores computed embeddings on disk so that unchanged documents are not re-embedded on the next indexing run. It saves API cost when you are re-indexing frequently with mostly unchanged content. For a one-time index build, it is not needed.

---

## Key Takeaways

- A RAG pipeline has two phases: indexing (offline, run once) and querying (per request) — keep them separate in code and infrastructure
- Use `RecursiveCharacterTextSplitter` at 512 chars / 64 overlap as the default starting point — measure with RAGAS before changing
- Use LCEL (`prompt | llm | StrOutputParser()`) — not deprecated `RetrievalQA.from_chain_type()`
- Use `from langchain_chroma import Chroma` — not `from langchain_community.vectorstores import Chroma`
- The grounding instruction ("answer ONLY from context") is the most critical line in the prompt
- Always use MMR retrieval instead of plain similarity to avoid returning 5 near-duplicate chunks
- Test both in-scope and out-of-scope queries before deployment — the "I do not know" path is as important as the happy path
- Never re-index on API startup — load the persisted vector store (under 1 second); never rebuild it

---

## What to Learn Next

- **Architecture decisions behind this code** → [RAG Architecture Guide](/blog/rag-architecture-guide/)
- **Chunking deep dive** → [RAG Chunking Strategies](/blog/rag-chunking-strategies/)
- **Measure if it works** → [RAG Evaluation with RAGAS](/blog/rag-evaluation/)
- **Add hybrid search** → [Hybrid Search RAG](/blog/hybrid-search-rag/)
