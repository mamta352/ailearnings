---
title: "RAG System Architecture: From Prototype to Production (2026)"
description: "Prototype RAG != production RAG. Design the full system — ingestion pipeline, vector store, retrieval layer, reranking, and observability."
date: "2026-03-10"
slug: "rag-system-architecture"
author: "Mamta Chauhan"
authorTitle: "Content Creator and AI Enthusiast"
level: "advanced"
time: "15"
keywords: ["RAG architecture", "RAG system design", "production RAG", "advanced RAG patterns"]
---

# RAG System Architecture: From Prototype to Production

Your first RAG demo works. You retrieve a few chunks, stuff them in a prompt, and the model answers correctly. Then you put it in front of real users.

Suddenly 20% of queries return wrong or hallucinated answers. Latency spikes under load. The model ignores perfectly good retrieved chunks. Error codes and product names come back with irrelevant results.

This is the gap between naive RAG and production RAG. This guide covers the full architecture — three generations of RAG design, hybrid retrieval, query transformation, reranking, and evaluation — with working code for each component.

---

## Three Generations of RAG Architecture

### Naive RAG

Simple retrieve → generate. Works for demos, fails in production.

```
User Query → Embed → Top-K Vector Search → Stuff into Prompt → LLM → Answer
```

Problems: poor retrieval precision, no query understanding, context window overflow.

### Advanced RAG

Pre-retrieval and post-retrieval processing to improve quality.

```
User Query → Query Transform → Hybrid Search → Reranking → Compression → LLM
```

### Modular RAG

Fully composable pipeline with specialized components for routing, fusion, iteration, and feedback. Each component is independently replaceable.

---

## Learning Objectives

- Understand the three generations of RAG architecture
- Design a production-grade RAG indexing pipeline
- Implement hybrid search (BM25 + vector)
- Apply reranking to improve retrieval precision
- Handle multi-document and multi-modal sources
- Monitor and debug RAG quality

---

## Indexing Pipeline

The indexing pipeline runs offline, preparing documents for fast retrieval.

```python
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

def build_index(source_dir: str, persist_dir: str) -> Chroma:
    all_docs = []

    # 1. Load documents from multiple sources
    for pdf_path in Path(source_dir).glob("**/*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        # Add source metadata
        for d in docs:
            d.metadata["source_type"] = "pdf"
            d.metadata["filename"] = pdf_path.name
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} raw documents")

    # 2. Clean and split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=64
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks")

    # 3. Embed and store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"Indexed {vectorstore._collection.count()} chunks")
    return vectorstore

vectorstore = build_index("./documents", "./chroma_db")
```

---

## Hybrid Search: BM25 + Vector

Combining keyword search (BM25) with semantic search dramatically improves retrieval, especially for specific terms, names, and codes.

```bash
pip install rank_bm25 langchain
```

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Vector retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# BM25 keyword retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10

# Ensemble: weighted combination
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],  # 40% keyword, 60% semantic
)

results = hybrid_retriever.invoke("What is the rate limit for the API?")
```

**When hybrid search helps most:**
- Queries containing specific model names, version numbers, error codes
- Domain-specific jargon that embeddings haven't seen in training
- Short queries (2–3 words) where semantic search lacks context

---

## Query Transformation

Naive RAG passes the raw user query directly. Advanced RAG transforms queries for better retrieval.

### Query Rewriting

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the following user question to be more specific and retrieval-friendly.
Expand abbreviations, add context, and make it a clear factual query.

Original question: {question}

Rewritten question:""")

rewrite_chain = rewrite_prompt | llm

original = "how do I set up auth?"
rewritten = rewrite_chain.invoke({"question": original})
print(rewritten.content)
# → "What are the steps to configure authentication and authorization in [system name]?"
```

### Multi-Query Retrieval

Generate multiple phrasings of the same question and merge results:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_retriever,
    llm=llm,
)

# Internally generates 3–5 variations of the query and merges unique results
results = multi_query_retriever.invoke("Tell me about rate limits")
```

### Step-Back Prompting

For complex questions, first ask a more general "step-back" question:

```python
step_back_prompt = ChatPromptTemplate.from_template("""
You are an expert at transforming specific questions into more general ones.
Given the specific question below, generate a more general question that
would help gather background information needed to answer it.

Specific question: {question}
General step-back question:""")
```

---

## Reranking

After initial retrieval (high recall), reranking reorders results for precision. A cross-encoder model scores each (query, document) pair.

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, documents: list, top_n: int = 4) -> list:
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )
    return [doc for _, doc in ranked[:top_n]]


# Retrieve more (higher recall), then rerank to top 4 (higher precision)
initial_docs = hybrid_retriever.invoke(query)  # top 10–20
reranked_docs = rerank(query, initial_docs, top_n=4)
```

---

## Context Compression

When retrieved chunks contain irrelevant content mixed with relevant content, extract only the relevant parts:

```python
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever,
)

# Returns only the parts of each document relevant to the query
compressed_docs = compression_retriever.invoke("What is the timeout setting?")
```

---

## Full Advanced RAG Pipeline

```python
class AdvancedRAG:
    def __init__(self, vectorstore, chunks):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Hybrid retriever
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 15
        vector = vectorstore.as_retriever(search_kwargs={"k": 15})
        self.retriever = EnsembleRetriever(
            retrievers=[bm25, vector], weights=[0.3, 0.7]
        )

        # Answer prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer the question using only the provided context.
Cite specific passages when possible. If the answer isn't in the context, say so.

Context:
{context}"""),
            ("human", "{question}"),
        ])

    def _rerank(self, query: str, docs: list, top_n: int = 5) -> list:
        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)
        return [d for _, d in sorted(zip(scores, docs), reverse=True)[:top_n]]

    def query(self, question: str) -> dict:
        # 1. Retrieve (high recall)
        docs = self.retriever.invoke(question)

        # 2. Rerank (high precision)
        top_docs = self._rerank(question, docs, top_n=5)

        # 3. Format context
        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('filename', '?')}]\n{d.page_content}"
            for d in top_docs
        )

        # 4. Generate answer
        response = (self.prompt | self.llm).invoke({
            "question": question,
            "context": context,
        })

        return {
            "answer": response.content,
            "sources": [d.metadata.get("filename") for d in top_docs],
            "chunks_retrieved": len(docs),
        }
```

---

## Evaluation Framework

```python
def evaluate_rag(rag: AdvancedRAG, test_cases: list[dict]) -> dict:
    """
    test_cases: [{"question": "...", "expected_keywords": ["...", "..."]}]
    """
    results = []
    for tc in test_cases:
        result = rag.query(tc["question"])
        answer = result["answer"].lower()

        keyword_hits = sum(
            1 for kw in tc["expected_keywords"] if kw.lower() in answer
        )
        score = keyword_hits / len(tc["expected_keywords"])
        results.append(score)

    return {
        "avg_score": sum(results) / len(results),
        "pass_rate": sum(1 for r in results if r >= 0.5) / len(results),
    }
```

---

## Troubleshooting

**Retrieved chunks are all from the same document**
→ Use MMR retrieval or add diversity constraints to the retriever.

**Model hallucinates despite having context**
→ Strengthen the system prompt: "Answer ONLY based on the provided context. If unsure, say you do not have that information." Consider adding a faithfulness check.

**Very slow query latency**
→ Cache embeddings for repeated queries. Use a faster embedding model (BGE-small). Pre-compute BM25 index offline.

---

## Frequently Asked Questions

**How is Advanced RAG different from a basic chatbot?**
Basic chatbots rely on LLM training data only. Advanced RAG grounds every answer in your specific documents — enabling accurate, citation-backed responses on proprietary, recent, or domain-specific information that the model was never trained on.

**Should I use LangChain or build my own pipeline?**
LangChain for fast prototyping — it gives you document loaders, splitters, retrievers, and chains in minutes. Custom pipeline for production — more control, fewer dependencies, easier debugging. Many teams prototype in LangChain and rewrite the retrieval layer in plain Python before launch.

**What is the difference between naive RAG and advanced RAG in practice?**
Naive RAG passes the raw query directly to vector search, returns the top-k chunks, and passes them to the LLM. Advanced RAG adds query transformation (rewriting, multi-query expansion), hybrid search (BM25 + vector), and reranking. In practice, the most impactful single upgrade is adding a cross-encoder reranker — it consistently improves answer quality with minimal added latency.

**How do I debug retrieval failures?**
First, print the retrieved chunks for failing queries. If the right chunk is not in the top 5, the problem is retrieval — try smaller chunk size, increase k, add hybrid search, or add a reranker. If the right chunk is present but the LLM ignores it, the problem is generation — strengthen the grounding instruction in the prompt.

**What is multi-query retrieval and when should I use it?**
Multi-query generates 3–5 rephrased versions of the original question and merges the retrieval results. This helps when a single phrasing misses relevant chunks due to vocabulary mismatch. Use it for complex or ambiguous queries. It adds one extra LLM call per query to generate the rephrasings.

**How does step-back prompting improve RAG?**
Step-back prompting generates a more general version of the original question ("What are the principles behind X?" instead of "What is the value of X in version 2.3?") and retrieves context for that general question first. This gives the LLM background context that makes specific answers more accurate.

**What reranker model should I use?**
`cross-encoder/ms-marco-MiniLM-L-6-v2` is the standard default — ~80MB, runs on CPU, ~50ms for 20 candidates. For higher quality, use `cross-encoder/ms-marco-electra-base` (~200MB). For a managed API option, Cohere Rerank requires no local model and consistently outperforms MiniLM at a cost of ~$0.001 per 1,000 documents ranked.

---

## Key Takeaways

- Naive RAG works for demos; production requires query transformation, hybrid retrieval, and reranking
- The indexing pipeline (load → chunk → embed → store) should run offline and independently from the query API
- Hybrid search (BM25 + vector, 40/60 weighting) improves retrieval especially for exact terms, codes, and domain-specific identifiers
- Query rewriting and multi-query retrieval improve recall for ambiguous or complex questions
- Cross-encoder reranking is the highest-ROI post-retrieval optimization — retrieve 15–20 candidates, rerank to top 5
- Use `EnsembleRetriever` for local hybrid search; use Weaviate or Pinecone native hybrid search for managed production deployments
- Evaluate retrieval and generation separately: keyword hit rate for retrieval, RAGAS faithfulness for generation
- The `AdvancedRAG` class pattern (hybrid retrieve → rerank → format context → LCEL chain) is the production-ready approach

---

## What to Learn Next

- **Full RAG pipeline code** → [LangChain RAG Tutorial](/blog/langchain-rag-tutorial/)
- **Measure retrieval and answer quality** → [RAG Evaluation with RAGAS](/blog/rag-evaluation/)
- **Vector database selection** → [Vector Database Guide](/blog/vector-database-guide/)
- **Context window optimization** → [Context Window RAG](/blog/context-window-rag/)
