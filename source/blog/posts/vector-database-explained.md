---
title: "Vector Databases: How They Power Search and RAG (2026)"
description: "Using a vector DB without understanding it? ANN indexing, HNSW graphs, metadata filtering."
date: "2026-03-13"
slug: "vector-database-explained"
keywords: ["vector database", "vector database explained", "FAISS", "Pinecone", "Weaviate", "semantic search database"]
author: "Mamta Chauhan"
authorTitle: "Content Creator and AI Enthusiast"
level: "beginner"
time: "12 min"
stack: ["Python", "FAISS", "ChromaDB", "Pinecone", "OpenAI"]
---

# Vector Databases Explained – FAISS, Pinecone, Weaviate

Traditional databases store structured data and search by exact match or range. A query for "documents about neural networks" returns nothing unless those exact words appear. Vector databases solve this with semantic search — finding content by meaning, not keywords. They are the foundation of every RAG system, recommendation engine, and semantic search application.

---

## What is a Vector Database

A vector database stores high-dimensional numerical vectors alongside their source content and metadata. Each vector represents the semantic meaning of a piece of data — a sentence, image, audio clip, or any other content that can be embedded.

When you search, the database converts your query to a vector and finds the stored vectors that are most geometrically similar. Geometrically similar vectors correspond to semantically similar content.

The key operation is **Approximate Nearest Neighbor (ANN) search**: find the K stored vectors closest to a query vector. "Approximate" is intentional — exact nearest neighbor search in high dimensions is too slow. ANN algorithms like HNSW and IVF trade a small amount of accuracy for large speed gains.

---

## Why Vector Databases Matter for Developers

Standard databases cannot answer "find documents similar to this query." They can only match on exact values or ranges. Vector databases unlock:

- **Semantic search** — Find documents by meaning, not keyword overlap
- **RAG retrieval** — Retrieve relevant context chunks for LLM prompts
- **Recommendation** — Find items similar to what a user liked or viewed
- **Duplicate detection** — Find near-duplicate content at scale
- **Anomaly detection** — Identify outliers far from all clusters

For AI applications specifically, vector databases are the retrieval layer in [RAG systems](/blog/rag-explained/). They store your document embeddings and return the most relevant passages when a user asks a question.

---

## How Vector Databases Work

### The Indexing Pipeline

```
Raw Data → Embedding Model → Dense Vector → Vector Store
"The quick brown fox" → [0.23, -0.14, 0.87, ...] → indexed
```

1. Pass text through an embedding model (OpenAI, sentence-transformers, etc.)
2. Store the resulting vector with the original text and metadata
3. Build an index structure for fast similarity search

### ANN Index Structures

**HNSW (Hierarchical Navigable Small World)** — The most widely used algorithm. Builds a graph of vectors connected to their neighbors. Very fast query times, good recall. Used by Qdrant, Weaviate, and Chroma.

**IVF (Inverted File Index)** — Partitions vectors into clusters. Searches only the most relevant clusters for a query. Used by FAISS. Good for very large datasets.

**Flat** — Brute-force exact search. Correct but slow at scale. Good for development or small datasets.

### The Query Pipeline

```
Query Text → Embed → ANN Search → Top-K Vectors → Return Text + Metadata
```

---

## Practical Examples

### FAISS (Local, In-Memory)

```python
import faiss
import numpy as np
from openai import OpenAI

client = OpenAI()

def embed(text: str) -> np.ndarray:
    res = client.embeddings.create(model="text-embedding-3-small", input=[text])
    return np.array(res.data[0].embedding, dtype="float32")

# Build index
documents = [
    "RAG combines retrieval with language model generation.",
    "Vector databases store embeddings for semantic search.",
    "Fine-tuning adapts a model to a specific domain.",
]

dim = 1536
index = faiss.IndexFlatL2(dim)
embeddings = np.array([embed(d) for d in documents])
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Query
query_vec = embed("How does retrieval work in AI?").reshape(1, -1)
faiss.normalize_L2(query_vec)
distances, indices = index.search(query_vec, k=2)

for i, idx in enumerate(indices[0]):
    print(f"[{distances[0][i]:.3f}] {documents[idx]}")
```

### Chroma (Persistent, Simple API)

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")
oai_ef = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection("docs", embedding_function=oai_ef)

# Add documents
collection.add(
    documents=["RAG retrieves relevant context before generation.",
               "FAISS provides fast approximate nearest neighbor search."],
    metadatas=[{"source": "rag-guide"}, {"source": "faiss-docs"}],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(query_texts=["how does retrieval work?"], n_results=2)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"[{meta['source']}] {doc[:100]}")
```

---

## Tools and Frameworks

| Database | Type | Best For |
|----------|------|----------|
| **FAISS** | Local library | Fast local search, research, prototyping |
| **Chroma** | Embedded / server | Local dev, small-medium production |
| **Pinecone** | Managed cloud | Production scale, no infrastructure management |
| **Weaviate** | Open-source / cloud | Hybrid search, GraphQL API, complex filtering |
| **Qdrant** | Open-source / cloud | High performance, rich filtering, Rust-based |
| **pgvector** | PostgreSQL extension | Teams already using PostgreSQL |
| **Redis** | In-memory / cloud | Low-latency search, existing Redis users |

**Choosing a vector database:**
- **Prototyping** → FAISS or Chroma
- **Small production** → Chroma or pgvector
- **Large scale, managed** → Pinecone or Weaviate Cloud
- **Self-hosted production** → Qdrant or Weaviate

---

## Common Mistakes

**Not normalizing vectors** — Cosine similarity requires unit-norm vectors. If you use L2 distance instead, normalization is less critical, but mixing approaches produces incorrect results. Be consistent.

**Embedding at query time with a different model** — The query embedding and document embeddings must use the same model. Mixing models produces nonsense results even if both models are high quality.

**Re-embedding already-indexed documents** — Once indexed, embeddings should not be recomputed unless the underlying text or model changes. Cache embeddings to avoid wasted API calls and cost.

**No metadata for filtering** — Storing vectors without metadata means you cannot filter results by document type, date, user, or any other attribute. Always store relevant metadata alongside vectors.

**Too many results (high K)** — Returning 20 chunks when the LLM context window only benefits from 3–5 dilutes the signal. Tune K based on your specific use case and model context size.

---

## Best Practices

- **Match embedding model to search domain** — Generic models work well for most text. For code, legal text, or medical content, consider domain-specific embedding models.
- **Use metadata filters** — Add source, date, category, and user context as metadata. Filter at search time to narrow results before semantic ranking.
- **Benchmark recall before deploying** — Measure what percentage of relevant documents are returned in the top-K for a representative set of queries. A recall below 70% means retrieval needs improvement.
- **Plan for index updates** — Documents change. Your indexing pipeline needs to handle additions, updates, and deletions. Most databases support upsert operations.
- **Monitor query latency** — Vector search should return in under 100ms for most applications. If it does not, tune the index type, reduce dimensions, or switch to a managed service.

---

## Key Takeaways

- Vector databases find content by semantic meaning, not keyword matching, by storing and searching dense numerical vectors
- The core operation is Approximate Nearest Neighbor (ANN) search, which trades a small amount of recall for millisecond query times at scale
- HNSW is the dominant ANN algorithm used in production vector databases due to its excellent recall-latency tradeoff
- You must use the same embedding model for both indexing and querying — mixing models produces meaningless similarity scores
- Always store metadata alongside vectors to enable structured filtering combined with semantic search
- FAISS and ChromaDB are the right tools for local development and prototyping; Pinecone and Qdrant serve production scale
- Normalizing vectors before storage ensures consistent cosine similarity computation across all queries
- Vector databases power RAG retrieval, semantic search, recommendation systems, and duplicate detection at production scale

---

## FAQ

**What is the difference between a vector database and a vector library like FAISS?**
FAISS is an indexing library — it handles ANN math only with no persistence, metadata, or API. A vector database like Chroma, Qdrant, or Pinecone adds persistence, metadata storage, filtering, and an API layer. Use FAISS when you need to embed search inside a larger system you control; use a database for production applications.

**How do vector databases handle updates and deletions?**
Most vector databases support upsert operations — insert if the ID does not exist, update if it does. Deletion is supported but may leave tombstone records that affect index quality over time. For high-churn datasets, periodic index rebuilds maintain search quality.

**Can I run a vector database without cloud infrastructure?**
Yes. ChromaDB runs fully in-process on your local machine with no external dependencies. Qdrant and Weaviate can be self-hosted via Docker. pgvector runs inside your existing PostgreSQL instance. Cloud is optional for all major vector databases.

**Do vector databases support transactions?**
No. Vector databases do not have ACID transactions, foreign keys, or joins. They are purpose-built for approximate similarity search, not transactional workloads. For data that requires transactional guarantees, keep it in a relational database and use vector search as an augmentation layer.

**What is the difference between HNSW and IVF indexing?**
HNSW builds a graph structure incrementally during insertion, delivering excellent query latency and good recall at the cost of higher memory usage. IVF partitions the vector space into clusters using k-means training, uses less memory, but requires a training phase and degrades if many vectors are added after training. HNSW is the better default for dynamic datasets.

**How does a vector database fit into a RAG system?**
In RAG, the vector database stores embeddings of your knowledge base documents. At query time, the user question is embedded and used to retrieve the most semantically relevant document chunks. These chunks are injected into the LLM prompt as context, enabling the model to answer questions grounded in your specific knowledge.

**What is metadata filtering in a vector database?**
Metadata filtering lets you combine vector similarity search with structured predicates — for example, "find the 5 most similar documents, but only from 2025 and only in the legal category." Most vector databases support pre-filtering (applied before ANN search) and post-filtering (applied after). Pre-filtering is more accurate for highly selective filters.

---

## What to Learn Next

- [Vector Databases Guide: Setup and Query in 30 Minutes](/blog/vector-database-guide/)
- [Embeddings Explained: How Text Becomes Searchable Vectors](/blog/embeddings-explained/)
- [Vector Database Comparison: Pinecone vs Qdrant vs Weaviate](/blog/vector-database-comparison/)
- [Build a RAG App from Scratch](/blog/build-rag-app/)
- [Semantic Search with Vector Databases](/blog/semantic-search-vector/)
