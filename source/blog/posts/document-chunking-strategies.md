---
title: "Text Chunking for RAG: Stop Losing Context in Splits (2026)"
description: "Bad chunks ruin good retrieval. Compare fixed, semantic, and hierarchical chunking — with LangChain splitter benchmarks and chunk size test code."
date: "2026-03-10"
slug: "document-chunking-strategies"
author: "Mamta Chauhan"
level: "intermediate"
time: "14"
keywords: ["document chunking RAG", "text chunking strategies", "RAG chunking", "chunk size for RAG", "langchain text splitter", "semantic chunking", "hierarchical chunking"]
---

You built a RAG pipeline, loaded your documents, and the retrieval still returns garbage. The LLM hallucinates or gives partial answers. You tune the prompt. Same result.

The problem is almost always the chunks.

Chunking is the most impactful and least discussed part of RAG. A bad split cuts a sentence in half. A too-large chunk buries the relevant sentence under irrelevant context. A too-small chunk retrieves the answer but strips the explanation the LLM needs to respond correctly.

This guide covers five chunking strategies, when to use each, a decision framework, and code to benchmark them against your own documents.

---

## What Is Chunking and Why It Matters

In a RAG pipeline, documents are split into smaller pieces (chunks), embedded, and stored in a vector database. At query time, the top-k most semantically similar chunks are retrieved and passed to the LLM.

The chunk is the unit of retrieval. The quality of every answer depends on whether the right chunk gets retrieved.

**Too large:** Embedding a 2,000-character chunk produces a single vector that averages over many ideas. Retrieval pulls the chunk because it partially matches — but the LLM gets buried in irrelevant context.

**Too small:** A 50-character chunk may contain exactly one fact. But the LLM needs context around that fact to generate a complete answer.

**Bad boundaries:** Splitting mid-sentence produces fragments. Embeddings of fragments are noisy and retrieve poorly.

The goal: **chunks that are semantically self-contained and as compact as possible while still answering a complete thought.**

---

## Strategy Comparison at a Glance

| Strategy | Quality | Speed | Cost | Best For |
|---|---|---|---|---|
| Fixed-size (char) | Baseline | Fast | Free | Simple prose, baseline testing |
| Token-based | Baseline | Fast | Free | Strict token-budget pipelines |
| Sentence-level | Good | Fast | Free | FAQs, news, support docs |
| Semantic | Best | Slow | Paid (embeddings) | Long docs with topic shifts |
| Parent-child | Best | Medium | Free | Tech docs, high-precision recall |

---

## Strategy 1: Fixed-Size Chunking

Split by character or token count with overlap. The simplest strategy and a reliable baseline.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# RecursiveCharacterTextSplitter tries delimiters in order:
# \n\n → \n → . → space → ""
# This respects paragraph and sentence boundaries better than CharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,         # ~12% overlap — prevents losing info at boundaries
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    length_function=len,
)

chunks = splitter.split_text(text)
print(f"Chunks: {len(chunks)}, avg len: {sum(len(c) for c in chunks) // len(chunks)} chars")
```

**Recommended sizes by content type:**

| Content Type | Chunk Size | Overlap |
|---|---|---|
| Dense technical docs | 256–512 chars | 50–80 chars |
| General articles/prose | 512–1024 chars | 80–128 chars |
| Long narratives / books | 1024–2048 chars | 200–256 chars |
| FAQ / short answers | 128–256 chars | 20–30 chars |

**When to use:** Start here. Works well for clean, structured prose. Use as your benchmark before trying more expensive strategies.

**Limitation:** Ignores semantic boundaries. Can split in the middle of a code block, table, or multi-sentence explanation.

---

## Strategy 2: Token-Based Chunking

LLMs operate on tokens, not characters. One token is roughly 4 characters for English text — but code, URLs, and non-ASCII content tokenize differently. Use `tiktoken` for precise token-budget control.

```bash
pip install tiktoken langchain-openai
```

```python
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_by_tokens(
    text: str,
    model: str = "gpt-4o",
    chunk_size: int = 256,
    overlap: int = 32,
) -> list[str]:
    enc = tiktoken.encoding_for_model(model)

    def token_length(s: str) -> int:
        return len(enc.encode(s))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=token_length,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)


chunks = chunk_by_tokens(document, chunk_size=256, overlap=32)
enc = tiktoken.encoding_for_model("gpt-4o")
avg_tokens = sum(len(enc.encode(c)) for c in chunks) // len(chunks)
print(f"{len(chunks)} chunks, avg {avg_tokens} tokens each")
```

**When to use:** When you are passing chunks directly to a model with a strict context window and need to guarantee no chunk exceeds N tokens. Also useful when calculating exact retrieval costs.

---

## Strategy 3: Sentence-Level Chunking

Group N complete sentences per chunk. Preserves semantic units better than character splitting.

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def chunk_by_sentences(
    text: str,
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
) -> list[str]:
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    chunks = []
    step = sentences_per_chunk - overlap_sentences
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk)

    return chunks


chunks = chunk_by_sentences(article_text, sentences_per_chunk=5, overlap_sentences=1)
```

**Simpler alternative using NLTK:**

```python
import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

def chunk_sentences_nltk(text: str, n: int = 5) -> list[str]:
    sentences = sent_tokenize(text)
    return [" ".join(sentences[i : i + n]) for i in range(0, len(sentences), n)]
```

**When to use:** News articles, FAQ pages, support documentation — any content where individual sentences carry distinct meaning. Outperforms fixed-size on Q&A retrieval benchmarks for short-answer content.

---

## Strategy 4: Semantic Chunking

Split where meaning changes, not where character count hits a limit. Uses embedding similarity between adjacent sentences to detect topic shifts.

```bash
pip install langchain-experimental langchain-openai
```

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# "percentile" mode: split when cosine distance between adjacent sentences
# exceeds the Nth percentile of all inter-sentence distances in the document.
# Higher percentile = fewer, larger chunks.
chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",   # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=90,
)

chunks = chunker.split_text(long_document)
print(f"Semantic chunks: {len(chunks)}")
for i, c in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ({len(c)} chars) ---\n{c[:200]}...")
```

**Threshold type guide:**

| Type | Behavior | Use When |
|---|---|---|
| `percentile` | Split at top-N% similarity drops | Most documents — reliable default |
| `standard_deviation` | Split beyond N std devs from mean | Uniform-structure docs (legal, academic) |
| `interquartile` | Uses IQR, robust to outliers | Mixed-topic docs with occasional off-topic sections |

**When to use:** Long documents with clearly distinct sections — annual reports, textbooks, multi-topic documentation. When retrieval quality matters more than speed or cost.

**Limitation:** Each sentence requires an embedding call. A 10,000-word document may need 200+ API calls. Cache results or use a local embedding model (e.g., `sentence-transformers`) to reduce cost.

---

## Strategy 5: Parent-Child (Hierarchical) Chunking

Store small child chunks in the vector index for precise retrieval, but return the larger parent chunk to the LLM for rich context. Best of both worlds.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Child: small, precise for matching
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
)

# Parent: larger, context-rich for LLM response
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents)

# Returns parent chunks even though child chunks matched the query
results = retriever.invoke("What is the authentication timeout?")
print(f"Retrieved {len(results)} parent chunks")
for r in results:
    print(r.page_content[:200])
```

**Full RAG chain with parent-child retriever:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context provided. If not in context, say you do not know.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("What is the authentication timeout?")
print(answer)
```

**When to use:** Technical documentation, API references, knowledge bases — anywhere you need both high retrieval precision and comprehensive context in the answer.

---

## Handling Special Content Types

### Code Blocks

Use language-aware splitters that respect function and class boundaries:

```python
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100,
)

markdown_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=512,
    chunk_overlap=50,
)

python_chunks = python_splitter.split_text(python_source_code)
```

### PDFs

PDFs often produce garbled text with hyphenated line breaks and multiple blank lines. Clean before chunking:

```python
from langchain_community.document_loaders import PyPDFLoader
import re

def clean_pdf_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse extra blank lines
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)    # rejoin hyphenated words
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)     # join soft line breaks
    return text.strip()

loader = PyPDFLoader("document.pdf")
pages = loader.load()
for page in pages:
    page.page_content = clean_pdf_text(page.page_content)
```

### Tables

Tables lose structure when extracted as plain text. Convert to Markdown before chunking so the LLM can parse them:

```python
import pdfplumber

def extract_tables_as_markdown(pdf_path: str) -> list[str]:
    tables_md = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table:
                    continue
                header = "| " + " | ".join(str(c or "") for c in table[0]) + " |"
                divider = "| " + " | ".join("---" for _ in table[0]) + " |"
                rows = [
                    "| " + " | ".join(str(c or "") for c in row) + " |"
                    for row in table[1:]
                ]
                tables_md.append("\n".join([header, divider] + rows))
    return tables_md
```

---

## Choosing a Strategy: Decision Framework

```
Is your document structured with clear paragraphs?
├── Yes → RecursiveCharacterTextSplitter (512 chars, 64 overlap)
└── No  → Does meaning shift gradually across the document?
          ├── Yes → SemanticChunker (percentile mode, threshold=90)
          └── No  → Do you need precise token budgeting?
                    ├── Yes → Token-based chunking (256 tokens, 32 overlap)
                    └── No  → Sentence-based (5 sentences, 1 overlap)

For any strategy: does retrieval precision need to be very high?
└── Yes → Layer parent-child on top of your chosen strategy
```

---

## Benchmarking Your Chunking Strategy

Before choosing a production strategy, measure how well each one retrieves relevant content for your actual queries:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI

client = OpenAI()

def embed(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return np.array([d.embedding for d in response.data])

def evaluate_chunking(
    chunks: list[str],
    test_queries: list[tuple[str, str]],  # [(query, expected_substring), ...]
    top_k: int = 3,
) -> dict:
    chunk_embeddings = embed(chunks)

    hits = 0
    for query, expected in test_queries:
        q_emb = embed([query])
        sims = cosine_similarity(q_emb, chunk_embeddings)[0]
        top_chunks = [chunks[i] for i in np.argsort(sims)[::-1][:top_k]]

        if any(expected.lower() in c.lower() for c in top_chunks):
            hits += 1

    return {
        "recall@3": round(hits / len(test_queries), 3),
        "num_chunks": len(chunks),
        "avg_chunk_len": int(np.mean([len(c) for c in chunks])),
    }


# Example usage
test_queries = [
    ("What is the authentication timeout?", "timeout"),
    ("How do I reset a password?", "reset"),
    ("Which roles have admin access?", "admin"),
]

# Compare strategies
results_fixed = evaluate_chunking(fixed_chunks, test_queries)
results_semantic = evaluate_chunking(semantic_chunks, test_queries)

print("Fixed-size :", results_fixed)
print("Semantic   :", results_semantic)
```

Run this benchmark on 20–30 representative queries from your domain before picking a strategy for production.

---

## Common Mistakes

**1. Not using overlap.**
Zero overlap means information at a chunk boundary is permanently lost. Always use 10–20% overlap for fixed and token-based strategies.

**2. Choosing chunk size without testing.**
A chunk size that works for a legal document fails for a technical FAQ. Always benchmark on a representative sample of your data.

**3. Splitting mid-code-block.**
If your documents contain code, use `RecursiveCharacterTextSplitter.from_language()`. Plain character splitting will cut function bodies mid-way.

**4. Ignoring PDF extraction quality.**
PyPDF2 and PyMuPDF produce different text quality for the same file. Always inspect extracted text before chunking. Headers, footers, and page numbers add noise — strip them.

**5. Using SemanticChunker without caching.**
Semantic chunking calls the embedding API once per sentence. For a 50-page document, that can cost $0.50 per run. Cache results or batch-embed before indexing.

**6. Skipping metadata.**
Chunks without source metadata are useless for citations and debugging. Always preserve `source`, `page`, `section`, and `chunk_index` as metadata.

```python
from langchain.schema import Document

# Always attach metadata to chunks
doc_chunks = [
    Document(
        page_content=chunk,
        metadata={
            "source": "annual-report-2025.pdf",
            "page": page_num,
            "chunk_index": i,
            "strategy": "recursive",
        },
    )
    for i, chunk in enumerate(chunks)
]
```

---

## Frequently Asked Questions

**What chunk size should I start with?**
Start with `RecursiveCharacterTextSplitter` at 512 characters with 64-character overlap. This works well for general prose. Benchmark it with `evaluate_chunking()` and adjust from there.

**Should I always use overlap?**
Yes, for fixed and token-based strategies. Use 10–20% of chunk size. Overlap prevents the case where a key sentence spans two chunks and appears in neither. Semantic and sentence-level chunking handle this naturally.

**Is semantic chunking worth the extra cost?**
For documents where topics shift gradually (reports, textbooks, research papers), yes — it typically improves recall by 15–30% compared to fixed-size. For uniform prose (news, FAQs), the improvement is smaller and may not justify the cost.

**How many chunks should I retrieve (top-k)?**
Start with k=3 to k=5. More chunks give the LLM more context but increase token cost and can dilute the answer if irrelevant chunks are included. Use your benchmark to find the sweet spot.

**What is parent-child chunking and when should I use it?**
Parent-child stores small child chunks (200 chars) in the vector index for precise semantic matching, but returns the larger parent chunk (1000 chars) to the LLM so it has enough context to answer. Use it when you need both high retrieval precision and complete answers — ideal for API docs, runbooks, and multi-section technical guides.

**Does chunking strategy matter if I use a reranker?**
Yes. A reranker improves ranking within the retrieved set but cannot recover a relevant chunk that was never retrieved in the first place. Fix chunking first, then add a reranker.

**How do I handle multilingual documents?**
Use `spacy` with the appropriate language model for sentence-level chunking. For semantic chunking, use a multilingual embedding model like `multilingual-e5-large` instead of `text-embedding-3-small`, which is English-optimized.

---

## Key Takeaways

- Chunking is the single highest-leverage optimization in a RAG pipeline — fix this before tuning prompts or switching models
- Start with `RecursiveCharacterTextSplitter` at 512 chars / 64 overlap; benchmark before changing
- Use token-based chunking when you need strict token budgets for the LLM context window
- Use sentence-level chunking for FAQ, news, and support content where sentences carry discrete meaning
- Use semantic chunking for long multi-topic documents where topic boundaries are important
- Use parent-child chunking when you need both precise retrieval and rich context in answers
- Always use overlap (10–20%) for fixed and token-based strategies
- Always attach metadata (source, page, chunk index) to every chunk
- For code blocks, use language-aware splitters; for PDFs, clean the text first; for tables, convert to Markdown
- Benchmark with `recall@3` on 20–30 representative queries before committing to a production strategy

---

## What to Learn Next

- **Build a full RAG pipeline** → [LangChain RAG Tutorial](/blog/langchain-rag-tutorial/)
- **Evaluate your RAG pipeline** → [RAG Evaluation with RAGAS](/blog/rag-evaluation/)
- **Vector databases for storing chunks** → [Vector Database Guide](/blog/vector-database-guide/)
- **Hybrid search (BM25 + semantic)** → [Hybrid Search RAG](/blog/hybrid-search-rag/)
