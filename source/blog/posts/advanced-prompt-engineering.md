---
title: "Advanced Prompt Engineering: ReAct, Tree of Thought & Agent Patterns (2026)"
description: "Master advanced prompt engineering techniques — ReAct, plan-and-solve, Tree of Thought, self-critique, meta-prompting, and prompt ensembling. Patterns used in production AI agents and multi-step workflows."
date: "2026-03-13"
slug: "advanced-prompt-engineering"
author: "Amit K Chauhan"
authorTitle: "Software Engineer & AI Builder"
updatedAt: "2026-03-21"
keywords: ["advanced prompt engineering", "ReAct prompting", "tree of thought", "prompt chaining", "self-critique prompting", "LLM agent patterns", "production prompting"]
---

# Advanced Prompt Engineering: ReAct, Tree of Thought & Agent Patterns (2026)

_Last updated: March 2026_

Your first AI feature works in the demo but breaks unpredictably in production — inconsistent formats, cascading errors, no way to debug which step failed. That is the gap between basic prompting and advanced prompt engineering. This guide covers the patterns that close it: ReAct agent loops, Tree of Thought reasoning, self-critique, meta-prompting, plan-and-solve, and prompt ensembling.

For foundational techniques (zero-shot, few-shot, chain-of-thought, constrained output), start with [17 Prompt Engineering Techniques](/blog/prompt-engineering-techniques/).

---

## What is Advanced Prompt Engineering

Advanced prompt engineering goes beyond single-prompt interactions. It is the discipline of composing prompts into multi-step workflows, giving models access to external tools, and designing systems that are reliable across thousands of requests.

Where basic prompting focuses on a single input/output pair, advanced prompting treats the LLM as a component in a pipeline. Each prompt has defined inputs, outputs, and contracts with the steps around it. You can test each step independently, version each prompt separately, and trace failures to specific nodes.

The result is a system that degrades gracefully, catches its own errors, and handles inputs that were not in your test set.

---

## Why Advanced Prompt Engineering Matters for Developers

Most real-world AI tasks are too complex for a single prompt. A user asks a question that requires searching a database, running a calculation, and summarizing a document. That is three operations — and each benefits from its own focused prompt.

Advanced techniques let you build AI systems that:
- Handle tasks too complex for one prompt
- Catch and correct their own errors before returning a response
- Use external tools (search, code execution, APIs, databases)
- Maintain consistent quality across varied inputs
- Degrade gracefully when inputs are ambiguous or malformed
- Route different input types to specialized handling paths

For developers building AI features into products, these patterns are the bridge between a demo that works once and a feature that works every time.

---

## 9 Advanced Prompt Engineering Techniques

### Technique 18: ReAct Prompting

Interleave **Thought → Action → Observation** cycles. The model reasons about what to do, calls a tool, observes the result, then reasons again. This pattern is the backbone of every production AI agent.

```
Thought: I need to find the current population of Tokyo.
Action: search("Tokyo population 2026")
Observation: Tokyo's population is approximately 13.96 million (2026 estimate).
Thought: I now have the data needed to answer.
Final Answer: Tokyo has approximately 13.96 million residents as of 2026.
```

ReAct works with any LLM API — agent frameworks like LangChain handle the tool-calling loop for you, but the underlying pattern is model-agnostic. See the full implementation below.

---

### Technique 19: Plan-and-Solve

Ask the model to write a plan first, then execute it step by step. This separates planning from execution, dramatically reducing errors on complex multi-step tasks.

```
System: First write a numbered plan for how to solve this problem.
        Then execute each step of the plan in order.

User: Analyze this dataset and produce a quarterly revenue summary
      with trend analysis and anomalies flagged.
```

The model produces a plan like "1. Parse columns, 2. Aggregate by quarter, 3. Calculate QoQ growth, 4. Flag >20% deviations" before executing. This works better than one-shot because the planning phase forces the model to identify steps it might otherwise skip.

---

### Technique 20: Scratchpad Prompting

Give the model an explicit working area to think before committing to an answer.

```
Use a <scratchpad> section for your working and reasoning.
Then provide your final answer in <answer> tags.
Do not include the scratchpad content in the final answer.

Question: {user_question}
```

Scratchpad keeps intermediate reasoning visible and separate from the response. It is useful for debugging — you can inspect the scratchpad to see exactly where the model went wrong without exposing the reasoning noise to end users.

---

### Technique 21: Tree of Thought (ToT)

Explore multiple reasoning branches simultaneously, evaluate each, and select the best path. More powerful than linear CoT for complex planning and creative problem-solving.

```python
from openai import OpenAI

client = OpenAI()

def tree_of_thought(problem: str, n_branches: int = 3) -> str:
    # Step 1: Generate multiple solution approaches
    branches_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Generate {n_branches} distinct approaches to solve this problem. Number each approach."},
            {"role": "user", "content": problem}
        ],
        temperature=0.8
    )
    branches = branches_response.choices[0].message.content

    # Step 2: Evaluate and select best branch
    selection = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Evaluate each approach for correctness, efficiency, and completeness. Select the best one and execute it fully."},
            {"role": "user", "content": f"Problem: {problem}\n\nApproaches:\n{branches}"}
        ],
        temperature=0
    )
    return selection.choices[0].message.content
```

ToT is expensive (multiple API calls per query) — use it for tasks where quality is worth the latency cost: complex planning, architectural decisions, multi-constraint optimization.

---

### Technique 22: Step-Back Prompting

For complex questions, first ask a more general background question to surface relevant knowledge, then use that to answer the specific question.

```python
def step_back(specific_question: str) -> str:
    # Step 1: Generate a more general background question
    background_q = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "What general background knowledge or principle would help answer this specific question? State the background question."},
            {"role": "user", "content": specific_question}
        ],
        temperature=0
    ).choices[0].message.content

    # Step 2: Answer the background question
    background_a = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": background_q}],
        temperature=0
    ).choices[0].message.content

    # Step 3: Answer the original question using the background
    final = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Background knowledge:\n{background_a}\n\nUse this background to answer the specific question."},
            {"role": "user", "content": specific_question}
        ],
        temperature=0
    ).choices[0].message.content

    return final
```

Reduces errors on questions that require domain background the model might not foreground automatically, especially in science, law, and engineering domains.

---

### Technique 23: Prompt Chaining

Split complex tasks across multiple prompts where the output of one becomes the input of the next. More reliable than a single large prompt for multi-step workflows, and each step is independently testable.

The key advantage: when a chain fails, you can run each step in isolation to find exactly which prompt produced bad output. You cannot do that with a single mega-prompt.

See the full implementation in the [Practical Example](#practical-example) section below.

---

### Technique 24: Meta-Prompting

Ask the model to generate or improve a prompt for a given task.

```python
def meta_prompt(task_description: str) -> str:
    """Generate an optimized system prompt for a given task."""
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are a prompt engineering expert.
Write a production-quality system prompt for the task described.
Include: role definition, task instructions, output format specification,
constraints, and one or two examples if helpful.
Return only the system prompt — no explanation."""},
            {"role": "user", "content": f"Task: {task_description}"}
        ],
        temperature=0.3
    ).choices[0].message.content

# Example
prompt = meta_prompt("Python code reviewer that catches security issues and returns structured JSON findings")
print(prompt)
```

Meta-prompting is useful for rapid exploration when you are not sure where to start. Use the generated prompt as a starting point, then refine with your domain knowledge.

---

### Technique 25: Self-Critique

Ask the model to critique its own output, then improve it. This adds a self-correction loop that measurably improves quality for writing, code generation, and structured outputs.

```python
def self_critique(task: str, initial_input: str) -> str:
    # First pass — generate initial output
    initial = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"{task}\n\n{initial_input}"}],
        temperature=0.3
    ).choices[0].message.content

    # Critique pass — identify issues
    revised = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"{task}\n\n{initial_input}"},
            {"role": "assistant", "content": initial},
            {"role": "user", "content": "Review your response for accuracy, completeness, and clarity. Rewrite it with improvements. If the original was correct, repeat it unchanged."}
        ],
        temperature=0.2
    ).choices[0].message.content

    return revised
```

Run the critique step at slightly higher temperature (0.2–0.4) to introduce diversity. The rewrite step at temperature=0 disciplines it. Self-critique improves quality but does not guarantee correctness — do not use it as a factual verification layer.

---

### Technique 26: Prompt Ensembling

Run multiple differently-phrased prompts for the same task and aggregate the outputs. Reduces sensitivity to specific phrasing and improves robustness for high-stakes tasks.

```python
from collections import Counter

def ensemble_classify(text: str, label_options: list[str]) -> str:
    """Run three prompt variants and take the majority classification."""
    prompts = [
        f"Classify this text as {'/'.join(label_options)}. Return only the label.\n\nText: {text}",
        f"What category best describes this? Options: {', '.join(label_options)}. Answer with just the category.\n\nText: {text}",
        f"Label: {', '.join(label_options)}\nInput: {text}\nAnswer with exactly one label:"
    ]

    votes = []
    for prompt in prompts:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        votes.append(response.choices[0].message.content.strip())

    return Counter(votes).most_common(1)[0][0]
```

Ensembling is especially valuable for classification tasks where consistent labeling matters more than raw accuracy. It smooths over phrasing sensitivity without requiring prompt optimization.

---

## How Advanced Prompt Engineering Works

### Prompt Chaining

Prompt chaining splits a complex task into a sequence of simpler, focused prompts. The output of each step feeds the next. Think of it as a pipeline where each stage has a clear responsibility.

The key insight is that a model focused on one task performs better than one juggling three. A prompt that asks "extract facts, then summarize, then write a headline" produces worse results than three separate prompts doing each task independently.

### Conditional Routing

Not every input needs the same treatment. A classifier prompt categorizes inputs, then routes them to specialized handlers. This lets you tune each path independently without coupling them together.

### Self-Critique Loops

Ask the model to evaluate and improve its own output. The model first drafts an answer, then reviews it for accuracy or quality issues, then rewrites it addressing the identified problems. This pattern works best for writing tasks, code generation, and structured outputs where quality criteria can be stated explicitly.

### Tool Augmentation (ReAct Pattern)

The **ReAct** pattern interleaves reasoning steps with tool calls. The model thinks, acts, observes the result of the action, then thinks again. This grounds the model's reasoning in real-world data rather than its own memory.

---

## Practical Example

### Prompt Chaining Implementation

```python
from openai import OpenAI

client = OpenAI()

def chain_extract_summarize_headline(article: str) -> dict:
    """A three-step prompt chain."""

    # Step 1: Extract key facts
    facts_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract the 5 most important facts from the article. Return as a numbered list. Be specific and include numbers/dates where present."},
            {"role": "user", "content": article}
        ],
        temperature=0,
    )
    facts = facts_response.choices[0].message.content

    # Step 2: Generate a summary from those facts
    summary_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write a 2-paragraph executive summary based on the provided facts. Write for a non-technical audience."},
            {"role": "user", "content": f"Facts:\n{facts}"}
        ],
        temperature=0.3,
    )
    summary = summary_response.choices[0].message.content

    # Step 3: Write a headline
    headline_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write a compelling, accurate headline under 10 words."},
            {"role": "user", "content": f"Summary:\n{summary}"}
        ],
        temperature=0.5,
    )
    headline = headline_response.choices[0].message.content

    return {"facts": facts, "summary": summary, "headline": headline}
```

Each step is small, focused, and testable independently. Failures are easier to debug because you know exactly which step produced bad output.

### Tool-Augmented Prompting (ReAct Pattern)

```python
import json

def get_weather(city: str) -> str:
    """Mock weather tool — replace with real API in production."""
    data = {"tokyo": "12°C, cloudy", "london": "8°C, rainy", "paris": "15°C, sunny"}
    return data.get(city.lower(), f"No data available for {city}")

def calculator(expression: str) -> str:
    """Safe math expression evaluator."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

TOOLS = {
    "get_weather": get_weather,
    "calculator": calculator,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions for a city. Use when the user asks about weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Use for any arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

def run_react_agent(user_query: str, max_steps: int = 6) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them whenever needed."},
        {"role": "user", "content": user_query},
    ]

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content  # Final answer — no more tools needed

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = TOOLS.get(name, lambda **k: "Unknown tool")(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Max steps reached without a final answer."

# Test
print(run_react_agent("What is the temperature in Tokyo? Multiply it by 1000."))
```

---

## Real-World Applications

Advanced prompt engineering patterns power most production AI systems:

**RAG pipelines** — Use chaining to rewrite the user query, retrieve chunks, generate an answer, and optionally verify the answer cites sources correctly. Each step is a focused prompt.

**Code generation workflows** — Generate code, then run a separate critique step checking for bugs, then apply a final formatting step. The model catches more issues than single-shot generation.

**Document processing** — Extract structured data, validate it, transform it to the target schema, and generate a human-readable summary. Four focused prompts outperform one mega-prompt.

**Customer support routing** — A classifier prompt routes tickets to specialized handlers (billing, technical, general). Each handler has a prompt tuned for its domain.

**AI agents** — Every agent is a ReAct loop: think, use a tool, observe the result, think again. The pattern scales from single-tool agents to complex multi-step workflows.

---

## Common Mistakes Developers Make

1. **Treating chains as black boxes** — Each step in a chain should be independently testable. Build a test set for each prompt in isolation before wiring them together. If you cannot test step 2 without running step 1 first, your architecture needs adjustment.

2. **No fallback on tool failure** — Tools fail. APIs time out. Always design prompt chains with fallback paths when tool calls return errors or empty results. Return a graceful degraded response rather than propagating the failure.

3. **Inconsistent output formats between steps** — If step 1 outputs a bulleted list but step 2 expects JSON, the chain breaks on edge cases. Define explicit output schemas at each step and validate them before passing output to the next step.

4. **Over-relying on self-critique** — Self-critique improves quality but does not guarantee correctness. The model can confidently confirm incorrect information. Use it as a quality filter, not as a verification layer for factual accuracy.

5. **Too many hops in a single chain** — Long chains accumulate errors. Each step can drift slightly from the original intent. Keep chains to 3–5 steps where possible. Break very long workflows into separate, independently-tested sub-chains.

---

## Best Practices

- **Define contracts between steps** — Specify the exact input format each step expects and the exact output format it should produce. Treat each prompt like a typed function signature.
- **Log every step in production** — When a chain fails, you need to know which step produced bad output. Log inputs and outputs at every node — not just the final result.
- **Fail fast on malformed inputs** — Validate inputs before passing them into a chain. An early validation step is cheaper than discovering malformed data three steps later.
- **Version each prompt separately** — When a chain misbehaves, you need to know which prompt changed. Store each prompt as a named, versioned artifact in version control.
- **Test chains end-to-end and step-by-step** — Unit test individual steps with fixed inputs. Integration test the full chain. Both are necessary.
- **Cache expensive steps** — If step 1 is a slow web search, cache the result and allow re-running only steps 2 and 3 during debugging.

---

## FAQ

**When should I use prompt chaining vs. a single prompt?**
Use chaining when the task has multiple distinct phases, when format needs to change between phases, or when you need to test each phase independently. For simple single-step tasks, a single prompt is faster and cheaper.

**Does self-critique actually improve output quality?**
Yes, measurably, for writing and code tasks. Run the critique step with a slightly higher temperature and the generation with temperature=0. The critique step introduces diversity; the rewrite step disciplines it.

**How do I debug a chain when the output is wrong?**
Log every intermediate result. Then work backward from the failing step — feed that step's input in isolation and see if the output is wrong. This narrows the problem to a single prompt.

**Can I use different models for different steps in a chain?**
Yes, and often this is the right approach. Use a smaller, faster model for classification and routing steps. Reserve larger models for the generation steps that require the most capability.

**How do ReAct agents differ from prompt chaining?**
Chains have a fixed sequence of steps defined at build time. ReAct agents determine the sequence dynamically — the model decides what to do next based on the result of the previous step. Agents are more flexible but less predictable.

**When is Tree of Thought worth the extra cost?**
When the task involves creative problem-solving, architectural planning, or constraint satisfaction where the first approach the model tries is often not the best. For simple generation or extraction tasks, linear CoT is sufficient and cheaper.

**What is the best use case for meta-prompting?**
Starting from scratch in a new domain. Meta-prompting gives you a reasonable first prompt faster than writing from scratch. Always refine the generated prompt with your domain knowledge and test it against real examples.

**How do I know if prompt ensembling is helping?**
Run your benchmark test set with single-prompt and ensembled versions. If the ensemble improves accuracy by less than 3–5%, the cost is not justified. Ensembling is most useful for high-stakes classification where consistency matters more than speed.

---

## Further Reading

- [OpenAI Function Calling Documentation](https://platform.openai.com/docs/guides/function-calling)
- [LangChain LCEL Documentation](https://python.langchain.com/docs/concepts/lcel/)
- [ReAct: Synergizing Reasoning and Acting in Language Models (paper)](https://arxiv.org/abs/2210.03629)
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [DSPy: Programming — not prompting — Foundation Models](https://github.com/stanfordnlp/dspy)

---

## What to Learn Next

- [17 Prompt Engineering Techniques](/blog/prompt-engineering-techniques/) — foundational techniques that advanced patterns build on
- [Chain-of-Thought Prompting Explained](/blog/chain-of-thought-prompting/) — the reasoning technique at the heart of most advanced patterns
- [Build AI Agents Step-by-Step](/blog/build-ai-agents/) — put these techniques into a complete agent implementation
