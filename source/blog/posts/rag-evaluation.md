---
title: "RAG Evaluation: Stop Hallucinations Before Production (2026)"
description: "RAG pipeline shipping wrong answers? RAGAS catches them — faithfulness, relevancy, context precision and recall measured with copy-paste Python code. Includes CI/CD integration."
date: "2026-03-15"
slug: "rag-evaluation"
keywords: ["rag evaluation metrics", "ragas evaluation", "rag faithfulness", "rag answer relevancy", "llm evaluation rag", "context precision recall", "rag pipeline testing", "deepeval rag"]
author: "Mamta Chauhan"
authorTitle: "Content Creator and AI Enthusiast"
updatedAt: "2026-03-28"
---

# RAG Evaluation: Stop Hallucinations Before Production (2026)

You ran 10 test questions, the answers looked reasonable, and you shipped. Three days later, a user asked the same question you tested — and got a completely different, wrong answer. No monitoring. No baseline. No way to tell if last week was better or worse.

This is how most RAG systems reach production.

A RAG pipeline has two independent failure surfaces: **retrieval** (wrong chunks in the context) and **generation** (LLM ignores or misrepresents the chunks it was given). Treating the system as a black box and only reading answers manually makes it nearly impossible to diagnose which layer is broken — or whether a change you made helped or hurt.

Good RAG evaluation is not optional. It is the feedback loop that makes iterative improvement possible. This guide covers the standard evaluation framework (RAGAS), how to implement it end-to-end with LCEL, how to interpret results to drive fixes, and how to automate it in CI/CD.

---

## The Four Core Metrics

Each metric measures a different failure surface:

| Metric | What It Measures | Failure = |
|---|---|---|
| **Faithfulness** | Is the answer supported only by the retrieved context? | Hallucination |
| **Answer Relevancy** | Does the answer address the question asked? | Off-topic or vague response |
| **Context Precision** | Are the retrieved chunks relevant to the question? | Retrieval noise |
| **Context Recall** | Does the context contain everything needed to answer? | Missing information at retrieval |

**The key insight:** a low faithfulness score with high context precision means the LLM is the problem — strengthen the grounding prompt. A low context precision score means retrieval is the problem — tune the retriever. Never try to fix both simultaneously.

---

## When You Need Ground Truth Labels

Most RAGAS metrics use LLM-as-a-judge and require no human labels. The exception is context recall:

| Metric | Requires Ground Truth? |
|---|---|
| Faithfulness | No — LLM compares answer to context |
| Answer Relevancy | No — LLM scores relevance to question |
| Context Precision | No — LLM scores each retrieved chunk |
| Context Recall | **Yes** — LLM compares context to expected answer |

For initial evaluation, you can skip context recall and measure the other three with no labeling effort. Add ground truth labels when you want to measure retrieval completeness.

---

## Step 1: Install and Configure RAGAS

```bash
pip install ragas langchain langchain-openai langchain-community chromadb datasets
export OPENAI_API_KEY="sk-..."
```

---

## Step 2: Build a Test Dataset

Build evaluation questions from real user queries when possible. Synthetic questions reveal less than questions actual users have asked.

```python
# evaluation_dataset.py
evaluation_samples = [
    {
        "question": "What is the return policy for digital products?",
        "ground_truth": "Digital products are non-refundable once the license key has been activated. Physical products can be returned within 30 days of purchase.",
    },
    {
        "question": "How long does standard shipping take?",
        "ground_truth": "Standard shipping takes 5-7 business days for domestic orders and 10-14 business days for international orders.",
    },
    {
        "question": "Can I upgrade my plan mid-cycle?",
        "ground_truth": "Plan upgrades take effect immediately and are prorated for the remaining billing period. Downgrades take effect at the next billing cycle.",
    },
    {
        "question": "What payment methods are accepted?",
        "ground_truth": "We accept Visa, Mastercard, American Express, PayPal, and bank transfers for orders over $500.",
    },
    {
        "question": "Is there a free trial available?",
        "ground_truth": "Yes, a 14-day free trial is available for all plans. No credit card is required to start the trial.",
    },
    # Always include out-of-scope questions to test the "I do not know" path
    {
        "question": "What is the square root of 144?",
        "ground_truth": "I do not have enough information to answer that.",
    },
]

print(f"Evaluation dataset: {len(evaluation_samples)} samples")
```

**Minimum viable evaluation set:** 20 questions covering easy factual, multi-step reasoning, and out-of-scope queries. Production monitoring needs 100–200 questions.

---

## Step 3: Run the RAG Pipeline and Collect Results

Use LCEL syntax — `RetrievalQA.from_chain_type()` is deprecated in LangChain v0.2+.

```python
# run_pipeline.py
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load vector store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="documents"
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

SYSTEM_PROMPT = """Answer ONLY from the provided context.
If the context does not contain the answer, say exactly:
"I do not have enough information to answer that."

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# LCEL chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def run_pipeline_for_evaluation(samples: list) -> list:
    """Run RAG pipeline on all samples and collect answers + contexts."""
    results = []
    for sample in samples:
        # Retrieve source docs separately to pass to RAGAS
        source_docs = retriever.invoke(sample["question"])
        answer = rag_chain.invoke(sample["question"])

        results.append({
            "question":    sample["question"],
            "answer":      answer,
            "contexts":    [doc.page_content for doc in source_docs],
            "ground_truth": sample.get("ground_truth", ""),
        })
    return results


pipeline_results = run_pipeline_for_evaluation(evaluation_samples)
print(f"Collected {len(pipeline_results)} pipeline results")
```

---

## Step 4: Run RAGAS Evaluation

```python
# evaluate.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Convert pipeline results to RAGAS dataset format
eval_data = {
    "question":    [r["question"] for r in pipeline_results],
    "answer":      [r["answer"] for r in pipeline_results],
    "contexts":    [r["contexts"] for r in pipeline_results],
    "ground_truth": [r["ground_truth"] for r in pipeline_results],
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation — RAGAS uses LLM-as-judge internally (costs ~$0.01–0.05 per sample)
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(result)
# Example output:
# {'faithfulness': 0.91, 'answer_relevancy': 0.87,
#  'context_precision': 0.83, 'context_recall': 0.79}

# Per-question breakdown
df = result.to_pandas()
print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]].to_string())
```

**Cost of running RAGAS on 100 samples (gpt-4o-mini as judge):** approximately $0.50–$2.00 depending on document length. Use gpt-4o as judge when accuracy of the evaluation itself matters more than cost.

---

## Step 5: Diagnose Failures by Component

```python
def diagnose_failures(df) -> list:
    """
    Classify each question by failure type.
    Low score tells you which component to fix.
    """
    LOW = 0.7
    diagnoses = []

    for _, row in df.iterrows():
        issues = []

        if row["context_precision"] < LOW and row["context_recall"] < LOW:
            issues.append("RETRIEVAL_FAILURE: wrong chunks retrieved entirely — check embedding model or query")
        elif row["context_precision"] < LOW:
            issues.append("RETRIEVAL_NOISE: irrelevant chunks included — reduce k, add metadata filter, or use reranker")
        elif row["context_recall"] < LOW:
            issues.append("RETRIEVAL_INCOMPLETE: missing relevant chunks — increase k or improve chunking strategy")

        if row["faithfulness"] < LOW:
            issues.append("HALLUCINATION: answer not grounded in context — strengthen system prompt")

        if row["answer_relevancy"] < LOW:
            issues.append("OFF_TOPIC: answer does not address the question — check prompt or LLM temperature")

        diagnoses.append({
            "question": row["question"][:60],
            "issues":   issues if issues else ["PASS"],
            "scores": {
                "faithfulness": round(row["faithfulness"], 3),
                "relevancy":    round(row["answer_relevancy"], 3),
                "precision":    round(row["context_precision"], 3),
                "recall":       round(row["context_recall"], 3),
            }
        })

    return diagnoses


for d in diagnose_failures(df):
    if d["issues"] != ["PASS"]:
        print(f"\nQ: {d['question']}")
        print(f"Scores: {d['scores']}")
        print(f"Issues: {d['issues']}")
```

---

## Step 6: Custom LLM-as-Judge for Domain-Specific Quality

RAGAS metrics are general-purpose. For domain-specific requirements (citation accuracy, regulatory compliance, tone), implement custom judges:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

CITATION_JUDGE = ChatPromptTemplate.from_messages([
    ("system", "You are an expert evaluator. Score ONLY with a number from 0.0 to 1.0."),
    ("human", """Does the answer correctly cite sources from the context?

Answer: {answer}
Context: {context}

Scoring:
1.0 = All claims are attributed to specific sources
0.5 = Some citations present but incomplete
0.0 = No attribution or citations contradict context

Score:"""),
])

judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)  # use stronger model for judging
citation_chain = CITATION_JUDGE | judge_llm

def score_citation(answer: str, contexts: list[str]) -> float:
    formatted = "\n".join([f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts)])
    result = citation_chain.invoke({"answer": answer, "context": formatted})
    try:
        return float(result.content.strip())
    except ValueError:
        return 0.0

# Evaluate
for r in pipeline_results:
    score = score_citation(r["answer"], r["contexts"])
    print(f"Citation score: {score:.2f} | Q: {r['question'][:50]}")
```

---

## Step 7: Automate in CI/CD

Run evaluations on every pull request or nightly — catch regressions before they reach production.

```python
# eval_pipeline.py — run as part of CI/CD
import json
from datetime import datetime
from pathlib import Path

THRESHOLDS = {
    "faithfulness":     0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall":   0.75,
}

def run_and_save_evaluation(pipeline_results: list, output_dir: str = "./eval_results") -> dict:
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset = Dataset.from_dict({
        "question":    [r["question"] for r in pipeline_results],
        "answer":      [r["answer"] for r in pipeline_results],
        "contexts":    [r["contexts"] for r in pipeline_results],
        "ground_truth": [r["ground_truth"] for r in pipeline_results],
    })
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

    summary = {
        "timestamp":         timestamp,
        "sample_count":      len(pipeline_results),
        "faithfulness":      round(float(result["faithfulness"]), 4),
        "answer_relevancy":  round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
        "context_recall":    round(float(result["context_recall"]), 4),
    }

    # Save for trend tracking
    output_path = Path(output_dir) / f"eval_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved: {output_path}")

    # Fail CI if any metric drops below threshold
    failed = []
    for metric, threshold in THRESHOLDS.items():
        if summary[metric] < threshold:
            failed.append(f"{metric}={summary[metric]} < {threshold}")

    if failed:
        print(f"REGRESSION DETECTED: {failed}")
        raise SystemExit(1)   # non-zero exit code fails the CI pipeline

    print("All metrics within thresholds.")
    return summary


summary = run_and_save_evaluation(pipeline_results)
```

**GitHub Actions integration:**

```yaml
# .github/workflows/rag-eval.yml
name: RAG Evaluation
on: [pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install ragas langchain-openai chromadb datasets
      - run: python eval_pipeline.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## Interpreting Scores: What to Fix

| Metric | Good | Investigate | Fix |
|---|---|---|---|
| Faithfulness | ≥ 0.85 | 0.70–0.85 | < 0.70 — strengthen grounding prompt |
| Answer Relevancy | ≥ 0.80 | 0.65–0.80 | < 0.65 — rewrite system prompt |
| Context Precision | ≥ 0.75 | 0.60–0.75 | < 0.60 — add reranker or metadata filter |
| Context Recall | ≥ 0.75 | 0.60–0.75 | < 0.60 — increase k or fix chunking |

---

## RAGAS vs DeepEval vs TruLens

| Framework | Strengths | Best For |
|---|---|---|
| **RAGAS** | Simple API, 4 core metrics, no labels needed for 3/4 metrics | Getting started, general RAG evaluation |
| **DeepEval** | More metrics (G-Eval, hallucination, toxicity), CI/CD native, assertion-style tests | Teams that want pytest-style test assertions |
| **TruLens** | Real-time tracing, dashboard UI, feedback functions | Monitoring production RAG apps |
| **LangSmith** | Native LangChain integration, full chain tracing | Teams already on LangChain who want tracing + evals |

For most teams: start with RAGAS, graduate to TruLens for production monitoring once you have a baseline.

---

## Common Mistakes

**Evaluating only happy-path questions.** A system that refuses out-of-scope questions scores well on faithfulness. Include out-of-scope, adversarial, and ambiguous questions in your eval set to test the full behavior spectrum.

**Treating RAGAS scores as absolute truth.** RAGAS uses LLM-as-judge, which is imperfect. Periodically have a human review a sample of responses alongside RAGAS scores to calibrate. If the judge disagrees with humans consistently, switch evaluation models.

**Not version-controlling the evaluation dataset.** Update eval questions as your system improves, but track which eval version produced which scores. Results are only comparable within the same eval set.

**Skipping context recall because it requires labels.** Context recall is often where RAG systems fail most. Write 20 ground truth answers upfront — it takes an hour and reveals retrieval gaps that other metrics miss entirely.

**Running RAGAS with gpt-4o-mini as judge.** The judge model determines evaluation quality. Use gpt-4o for judging even if gpt-4o-mini generates your answers. A weak judge produces noisy scores that mislead decisions.

**Only running evaluations before launch.** Prod data distribution shifts over time. New document types, new user query patterns, and model updates can all degrade RAG quality silently. Run evaluations on a weekly sample of production queries.

---

## Frequently Asked Questions

**What is RAGAS?**
RAGAS (Retrieval Augmented Generation Assessment) is an open-source Python framework for evaluating RAG systems. It implements four core metrics — faithfulness, answer relevancy, context precision, and context recall — using an LLM-as-judge pattern that requires no human labels for three of the four metrics.

**Do I need human labels to use RAGAS?**
For faithfulness, answer relevancy, and context precision — no. RAGAS uses an LLM to score these automatically. For context recall, you need ground truth answers for each question. Ground truth can be written by a subject-matter expert or extracted directly from document content.

**How large should the evaluation dataset be?**
Twenty questions is the minimum for meaningful aggregate scores. One hundred to two hundred is the right target for production monitoring. Include a mix: simple factual queries, multi-step reasoning, out-of-scope questions, and queries that require combining information from multiple chunks.

**What faithfulness score should I target?**
Target 0.85 or higher for production. Scores above 0.90 indicate the grounding prompt is working well. Scores below 0.80 mean the LLM is regularly generating claims not supported by the retrieved context — the most common form of RAG hallucination.

**Can I use RAGAS with local models or Ollama?**
Yes. RAGAS supports any LangChain-compatible LLM. Replace the default OpenAI evaluator with Ollama: pass `llm=ChatOllama(model="llama3.1:8b")` to RAGAS. Smaller local models produce less reliable evaluation scores — for judging, models with 70B+ parameters are significantly more accurate than 7B–13B models.

**How much does running RAGAS evaluation cost?**
With gpt-4o-mini as judge, evaluating 100 samples costs roughly $0.50–$2.00 depending on document length. With gpt-4o as judge, expect $3–$10 for 100 samples. Run evaluations on a small representative set (20–30 samples) during development and the full set weekly in production.

**How do I integrate RAG evaluation into CI/CD?**
Run the eval script as a CI job on every pull request. Use a non-zero exit code when any metric drops below your threshold — this fails the build like a test failure. Store results as JSON artifacts for trend tracking. See the GitHub Actions example in the CI/CD section above.

---

## Key Takeaways

- RAG evaluation must measure **retrieval and generation separately** — a single end-to-end quality score cannot tell you which component is broken.
- **Faithfulness** detects hallucination. **Context Precision/Recall** detect retrieval problems. **Answer Relevancy** detects off-topic responses. Fix the right layer for each failure type.
- RAGAS requires no human labels for 3 of 4 metrics. **Start evaluating immediately** — do not wait until you have a labeled dataset.
- Use **gpt-4o as judge**, even if your pipeline uses gpt-4o-mini. A weak judge produces unreliable scores that mislead decisions.
- **Automate in CI/CD** with regression thresholds. Eval scores that are not tracked trend downward silently.
- Include **out-of-scope questions** in every eval set. Systems that only get tested on answerable questions have unknown behavior on the long tail.
- Production monitoring matters as much as pre-launch evaluation. **User query patterns shift** over time — weekly eval samples catch regressions before users do.

---

## What to Learn Next

- **Build the RAG pipeline being evaluated** → [Build a RAG App: Step-by-Step](/blog/build-rag-app/)
- **Improve retrieval with better chunking** → [Document Chunking Strategies](/blog/document-chunking-strategies/)
- **Add hybrid search to improve context recall** → [Hybrid Search for RAG](/blog/hybrid-search-rag/)
- **Understand the architecture being tested** → [RAG Architecture Guide](/blog/rag-architecture-guide/)
- **Evaluate more complex multi-document RAG** → [Multi-Document RAG](/blog/multi-document-rag/)
