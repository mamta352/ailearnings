---
title: "Prompt Engineering Examples for Real AI Applications"
description: "Practical prompt engineering examples for production AI use cases — classification, extraction, summarization, code review, RAG, and agent workflows."
date: "2026-03-15"
slug: "prompt-engineering-examples"
keywords: ["prompt engineering examples", "llm prompt examples", "prompt design", "few-shot examples", "production prompts"]
author: "Amit K Chauhan"
authorTitle: "Software Engineer & AI Builder"
updatedAt: "2026-03-15"
---

# Prompt Engineering Examples for Real AI Applications

Teams building AI features often struggle not with understanding prompting concepts, but with knowing how to write prompts for their specific use case. The gap between "I understand chain-of-thought" and "I know how to structure a prompt for our invoice extraction pipeline" is where most developers get stuck.

This post is a collection of battle-tested prompt patterns for real application categories. Each example is copy-paste runnable and includes the reasoning behind the design choices.

A common mistake I've seen in production: developers write prompts that work in the playground but fail in production because the test inputs were too clean. Real data is messy. The examples here account for that.

## Concept Overview

Prompt engineering examples serve two purposes. First, they demonstrate the pattern — what a well-structured prompt looks like for a given task type. Second, they serve as few-shot examples in your own prompts, showing the model what good output looks like.

The examples in this guide cover six high-value application categories:
1. Text classification
2. Structured data extraction
3. Document summarization
4. Code review and analysis
5. Retrieval-augmented question answering
6. Multi-step agent workflows

In practice, most production AI features fall into one of these categories or combine two of them.

## How It Works

![Architecture diagram](/assets/diagrams/prompt-engineering-examples-diagram-1.png)

The workflow is always the same: pick the simplest technique that fits the task, add format constraints, test against real (messy) data, and add negative instructions for failure patterns you observe.

## Implementation Example

### 1. Support Ticket Classification

This is one of the most common enterprise AI use cases. The challenge is not the classification itself — it is handling ambiguous tickets and maintaining consistent categories over time.

```python
from openai import OpenAI
import json

client = OpenAI()

CLASSIFIER_SYSTEM = """You are a support ticket classifier. Assign each ticket to exactly one primary category.

Categories:
- billing: charges, invoices, payments, refunds, subscriptions
- technical: bugs, errors, crashes, performance, API issues
- feature_request: asking for new capabilities or improvements
- account: login, passwords, 2FA, permissions, profile
- general: anything that doesn't fit above

Rules:
- If multiple categories apply, choose the most urgent one
- Billing issues take priority over technical issues when combined
- Return ONLY valid JSON, no explanation

Response schema:
{"category": string, "confidence": "high"|"medium"|"low", "secondary_category": string|null}"""

EXAMPLES = [
    ("My API key stopped working after I downgraded my plan.",
     '{"category": "billing", "confidence": "high", "secondary_category": "technical"}'),
    ("I think there\'s a bug — the export button does nothing when I click it.",
     '{"category": "technical", "confidence": "high", "secondary_category": null}'),
    ("Can you add dark mode to the dashboard?",
     '{"category": "feature_request", "confidence": "high", "secondary_category": null}'),
]

def classify_ticket(ticket: str) -> dict:
    messages = [{"role": "system", "content": CLASSIFIER_SYSTEM}]
    for user, assistant in EXAMPLES:
        messages += [
            {"role": "user", "content": f"Ticket: {user}"},
            {"role": "assistant", "content": assistant}
        ]
    messages.append({"role": "user", "content": f"Ticket: {ticket}"})

    r = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(r.choices[0].message.content)

# Test
print(classify_ticket("I was charged $99 but my plan should be $49. Also the invoices page is blank."))
# {"category": "billing", "confidence": "high", "secondary_category": "technical"}
```

### 2. Structured Data Extraction from Unstructured Text

Extraction is where format constraints are non-negotiable. If your downstream code expects a specific schema, the prompt must enforce it explicitly.

```python
EXTRACTION_SYSTEM = """Extract job posting information into structured JSON.

Rules:
- Use null for any field not explicitly mentioned in the text
- Normalize salary to annual USD when possible
- "remote" is true only if the posting explicitly states remote or work from home
- Do not infer or guess information not present

Schema:
{
  "company": string,
  "role": string,
  "location": string | null,
  "remote": boolean,
  "salary_min_usd": integer | null,
  "salary_max_usd": integer | null,
  "required_skills": [string],
  "seniority": "junior" | "mid" | "senior" | "staff" | "principal" | null
}"""

def extract_job(posting_text: str) -> dict:
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": posting_text}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(r.choices[0].message.content)

sample = """
Stripe is hiring a Senior Backend Engineer to join our Payments Infrastructure team
in San Francisco or New York. Salary range: $180k–$240k/year. We're looking for
engineers with 5+ years of Go or Rust experience, strong distributed systems background,
and experience with high-throughput data pipelines. This is an in-office role.
"""

print(json.dumps(extract_job(sample), indent=2))
```

### 3. Document Summarization with Audience Targeting

One thing many developers overlook: a generic "summarize this" prompt will produce generic summaries. Specifying the audience and purpose produces dramatically better output.

```python
SUMMARY_TEMPLATES = {
    "executive": """Summarize the following document for a senior executive.
- 3 bullet points maximum
- Focus on business impact and decisions required
- Avoid technical jargon
- Each bullet: one sentence
Do not add a header or label.""",

    "engineer": """Summarize the following technical document for a senior software engineer.
- Include architectural decisions and trade-offs
- Highlight implementation risks and dependencies
- 4–6 bullet points
- Technical precision over simplicity""",

    "sales": """Summarize the following for a sales representative preparing for a customer call.
- Focus on customer benefits and competitive advantages
- Avoid internal implementation details
- 3 bullet points with clear value statements"""
}

def summarize(document: str, audience: str = "executive") -> str:
    system = SUMMARY_TEMPLATES.get(audience, SUMMARY_TEMPLATES["executive"])
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": document}
        ],
        temperature=0.2
    )
    return r.choices[0].message.content
```

### 4. Code Review with Structured Findings

Code review is one of the highest-value AI use cases. The key is forcing structured output so findings can be sorted, filtered, and tracked programmatically.

```python
CODE_REVIEW_SYSTEM = """You are a senior software engineer specializing in Python security and code quality.

Review the provided code. Return findings in this exact JSON structure:
{
  "summary": string,
  "bugs": [
    {"line": integer, "description": string, "fix": string}
  ],
  "security_issues": [
    {"line": integer, "description": string, "severity": "low"|"medium"|"high"|"critical", "cwe": string|null}
  ],
  "performance_issues": [
    {"line": integer, "description": string, "impact": string}
  ],
  "code_quality": [
    {"line": integer, "description": string, "suggestion": string}
  ]
}

Rules:
- Use empty arrays [] for categories with no issues found
- Line numbers must be accurate
- Security findings take priority — never omit them
- "fix" and "suggestion" fields must contain actionable, concrete guidance"""

def review_code(code: str, language: str = "Python") -> dict:
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": CODE_REVIEW_SYSTEM},
            {"role": "user", "content": f"Language: {language}\n\nCode:\n```{language.lower()}\n{code}\n```"}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(r.choices[0].message.content)

vulnerable_code = """
import sqlite3

def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()
"""

findings = review_code(vulnerable_code)
for issue in findings.get("security_issues", []):
    print(f"[{issue['severity'].upper()}] Line {issue['line']}: {issue['description']}")
```

### 5. RAG Question Answering with Grounding

For retrieval-augmented generation, the most important prompt engineering decision is how to handle the case where the retrieved context does not contain the answer. The default — letting the model fall back to training knowledge — causes hallucinations that undermine user trust.

```python
RAG_SYSTEM = """You are a helpful assistant that answers questions based strictly on provided context.

Rules:
- Answer ONLY from the provided context
- If the context does not contain enough information, respond: "I don't have enough information in the provided sources to answer this question."
- Never use knowledge outside the provided context
- Cite the source number for every factual claim: [Source 1], [Source 2], etc.
- If sources contradict each other, note the discrepancy"""

def rag_answer(question: str, retrieved_chunks: list[dict]) -> str:
    # Format retrieved chunks with source labels
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[Source {i}] {chunk['source']}:\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0
    )
    return r.choices[0].message.content

# Example usage
chunks = [
    {"source": "docs/pricing.md", "text": "The Pro plan costs $49/month and includes 10 seats and 100GB storage."},
    {"source": "docs/billing.md", "text": "Billing occurs on the 1st of each month. Annual plans receive a 20% discount."}
]

answer = rag_answer("How much does the Pro plan cost annually?", chunks)
print(answer)
# The Pro plan costs $49/month [Source 1]. With an annual plan discount of 20% [Source 2],
# the annual cost would be $49 × 12 × 0.8 = $470.40/year.
```

## Best Practices

**Test with the messiest data you have.** Clean test inputs lead to false confidence. Pull 20 real examples from production logs — including edge cases, incomplete inputs, and unusual formatting — and build your test set from those.

**Define the failure mode explicitly.** What should the model do when input is ambiguous? When context is missing? When the answer is unknown? Every production prompt needs explicit instructions for these cases, not just the happy path.

**Use `response_format: json_object` for all structured extraction.** This enforces valid JSON at the API level and prevents markdown formatting from wrapping your JSON.

**Keep few-shot examples diverse.** If all your examples are clean, well-formatted inputs, the model will not generalize to messy real-world data. Include at least one example with unusual input.

**Log the actual prompt sent, not just the template.** When debugging production issues, you need to see the exact prompt with all variables filled in, not the template. Include prompt text in your observability pipeline.

## Common Mistakes

**Using the same prompt for all audience types.** A summarization prompt that works for engineers produces jargon-heavy output for sales teams. Audience targeting is a first-class design decision, not an afterthought.

**Not handling the "context gap" in RAG.** The most damaging RAG failure mode is the model filling gaps with hallucinated information. The fix is one explicit instruction: "If the context does not contain the answer, say so." Most teams forget this.

**Combining too many tasks in one prompt.** "Extract facts, summarize, classify, and suggest action items" is four tasks. Split them into a prompt chain. Each step is more accurate and independently testable.

**Assuming the model knows your domain vocabulary.** Abbreviations, internal product names, and industry jargon need to be defined or demonstrated in the prompt. Never assume the model shares your domain context.

## Summary

Production prompt engineering is applied pattern matching. The patterns in this guide — classification, extraction, summarization, code review, RAG, and chaining — cover the vast majority of real AI application use cases.

The underlying design principles are consistent across all of them: be explicit about format, handle edge cases in the prompt, test on messy real data, and use structured output APIs rather than hoping for correct JSON formatting.

## Related Articles

- [Prompt Engineering Guide for AI Developers](/blog/prompt-engineering-guide/)
- [26 Prompt Engineering Techniques Every AI Developer Should Know](/blog/prompt-engineering-techniques/)
- [Prompt Templates for AI Applications](/blog/prompt-templates/)
- [System Prompts: How to Control LLM Behavior](/blog/system-prompts-guide/)

## FAQ

**How many few-shot examples do I need for reliable classification?**
For well-defined categories, 2–3 examples per category is usually sufficient. More important than quantity is coverage — make sure examples cover ambiguous cases and edge cases, not just clear-cut inputs.

**Should I use GPT-4o or a smaller model for these tasks?**
Start with GPT-4o to get the prompt right, then test with smaller models like GPT-4o-mini. Many classification and extraction tasks run reliably on smaller models with a well-engineered prompt. Migration is easier once you have a test set.

**How do I handle multi-language inputs in extraction prompts?**
Add an instruction like "The input may be in any language. Always return the response in English." For entity extraction, specify: "Normalize names and terms to English when possible, preserve original spelling for proper nouns."

**What if the structured output schema changes over time?**
Version your prompts and schemas together. A schema change is a prompt change. Maintain a PROMPT_VERSION constant alongside your schema definition, and validate that model outputs match your expected schema in production.

**How do I improve accuracy without increasing prompt length?**
Replace vague instructions with specific examples. Two good examples are more effective than two paragraphs of instructions and use fewer tokens. Also check whether the failure is in the prompt or in the model's capability — some tasks require a more capable model, not a longer prompt.
