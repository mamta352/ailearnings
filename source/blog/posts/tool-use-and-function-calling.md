---
title: "Tool Use & Function Calling: LLMs That Do Real Work (2026)"
description: "Text generation is not enough. Implement OpenAI function calling and Anthropic tool use — query APIs, databases, and run code with typed JSON schemas."
date: "2026-03-10"
slug: "tool-use-and-function-calling"
author: "Mamta Chauhan"
authorTitle: "Content Creator and AI Enthusiast"
updatedAt: "2026-03-10"
level: "intermediate"
time: "15 min"
stack: ["Python", "OpenAI", "Anthropic"]
keywords: ["LLM function calling", "tool use LLM", "OpenAI function calling", "AI tool use"]
---

## Learning Objectives

- Define effective tool schemas that LLMs call reliably
- Handle single and parallel tool calls
- Design robust tool functions with proper error handling
- Build reliable multi-tool agentic pipelines
- Test and debug tool use

---

## What Is Function Calling?

Function calling (tool use) allows an LLM to:
1. Decide that an external tool is needed
2. Output a structured JSON call with arguments
3. Receive the tool's output
4. Incorporate the result into its answer

This is fundamentally different from asking the LLM to "write code" — the model outputs structured data you execute, not code.

```
User: "What's the weather in Tokyo?"
  ↓
LLM: { "tool": "get_weather", "args": { "city": "Tokyo" } }
  ↓
Code: runs get_weather("Tokyo") → "22°C, sunny"
  ↓
LLM: "The weather in Tokyo is 22°C and sunny."
```

---

## Defining Tools: Schema Best Practices

```python
from openai import OpenAI
client = OpenAI()

# ✅ Good tool definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": (
                "Search the product database for items matching a query. "
                "Use this when the user asks about product availability, pricing, or specifications. "
                "Returns a list of matching products with name, price, and stock status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Product search query (e.g., 'wireless headphones', 'laptop under $1000')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-20, default: 5)",
                        "default": 5,
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "books", "home", "all"],
                        "description": "Product category to filter by (default: 'all')",
                    },
                },
                "required": ["query"],
            },
        },
    }
]
```

**Schema design rules:**
1. **Name:** Use `snake_case`, be descriptive (`get_customer_order_status` not `status`)
2. **Description:** Explain WHEN to use the tool, not just what it does
3. **Parameters:** Describe expected format with examples. Use `enum` to constrain values
4. **Required:** Only mark parameters required if the tool truly can't work without them

---

## Basic Tool Call Handling

```python
import json

def execute_tool(name: str, args: dict) -> str:
    """Route tool call to the right function."""
    if name == "search_database":
        return search_database(**args)
    elif name == "get_weather":
        return get_weather(**args)
    elif name == "send_email":
        return send_email(**args)
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


def run_with_tools(user_message: str, max_iterations: int = 5) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when you need real-time data."},
        {"role": "user",   "content": user_message},
    ]

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        if response.choices[0].finish_reason == "stop":
            return msg.content  # No tool call, final answer

        # Handle tool calls
        for tool_call in msg.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
                result = execute_tool(tool_call.function.name, args)
            except json.JSONDecodeError:
                result = json.dumps({"error": "Invalid tool arguments"})
            except Exception as e:
                result = json.dumps({"error": str(e)})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Could not complete the request within the iteration limit."
```

---

## Parallel Tool Calls

GPT-4o can call multiple tools simultaneously when they're independent:

```python
def search_products(query: str) -> str:
    return json.dumps({"results": [{"name": f"Product for {query}", "price": 29.99}]})

def check_inventory(product_id: str) -> str:
    return json.dumps({"product_id": product_id, "in_stock": True, "quantity": 42})

def get_shipping_estimate(zip_code: str) -> str:
    return json.dumps({"zip": zip_code, "days": 3, "cost": 5.99})

multi_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search product catalog",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Check if a product is in stock",
            "parameters": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_shipping_estimate",
            "description": "Get shipping time estimate",
            "parameters": {"type": "object", "properties": {"zip_code": {"type": "string"}}, "required": ["zip_code"]},
        },
    },
]

# Model may call multiple tools at once
messages = [{"role": "user", "content": "Check if headphones are in stock and how long shipping to 94102 takes."}]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=multi_tools)

msg = response.choices[0].message
print(f"Tool calls: {len(msg.tool_calls)}")  # May be 2 parallel calls

messages.append(msg)
for tc in msg.tool_calls:
    args = json.loads(tc.function.arguments)
    result = {"search_products": search_products, "check_inventory": check_inventory,
              "get_shipping_estimate": get_shipping_estimate}.get(tc.function.name, lambda **k: "{}")(** args)
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

# Final answer
final = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(final.choices[0].message.content)
```

---

## Forcing Tool Use

```python
# Force a specific tool
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "search_database"}},
)

# Require any tool call (don't allow "I don't need a tool" response)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="required",
)

# Never use tools
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="none",
)
```

---

## Robust Tool Functions

Tool functions should handle errors gracefully and return structured responses:

```python
import requests
from typing import Any

def safe_tool(func):
    """Decorator that catches exceptions and returns error JSON."""
    def wrapper(**kwargs) -> str:
        try:
            result = func(**kwargs)
            return json.dumps({"success": True, "data": result})
        except ValueError as e:
            return json.dumps({"success": False, "error": f"Invalid input: {e}"})
        except requests.HTTPError as e:
            return json.dumps({"success": False, "error": f"API error: {e.response.status_code}"})
        except Exception as e:
            return json.dumps({"success": False, "error": f"Unexpected error: {type(e).__name__}"})
    return wrapper


@safe_tool
def lookup_stock_price(ticker: str) -> dict:
    """Look up current stock price for a ticker symbol."""
    if not ticker.isalpha() or len(ticker) > 5:
        raise ValueError(f"Invalid ticker symbol: {ticker}")

    # In production: use a real financial API
    prices = {"AAPL": 175.25, "GOOGL": 142.80, "MSFT": 415.30}
    if ticker not in prices:
        raise ValueError(f"Unknown ticker: {ticker}")

    return {"ticker": ticker, "price": prices[ticker], "currency": "USD"}
```

---

## Testing Tool Schemas

Always test that your schemas produce the correct tool calls:

```python
def test_tool_schema(user_message: str, expected_tool: str, expected_args: dict):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_message}],
        tools=tools,
    )

    msg = response.choices[0].message
    assert msg.tool_calls, f"Expected tool call but got: {msg.content}"

    call = msg.tool_calls[0]
    assert call.function.name == expected_tool, f"Expected {expected_tool}, got {call.function.name}"

    actual_args = json.loads(call.function.arguments)
    for key, value in expected_args.items():
        assert key in actual_args, f"Missing expected argument: {key}"
        if value is not None:
            assert actual_args[key] == value, f"Arg {key}: expected {value}, got {actual_args[key]}"

    print(f"✓ '{user_message}' → {expected_tool}({actual_args})")


# Test cases
test_tool_schema("Search for wireless headphones", "search_database", {"query": "wireless headphones"})
test_tool_schema("Find books in the electronics category", "search_database", {"category": "electronics"})
```

---

## Troubleshooting

**Model doesn't call the tool when expected**
- Improve the tool description — add "Use this when..." guidance
- Test with a more capable model (gpt-4o instead of gpt-4o-mini)
- Make the use case more explicit in the user message

**Tool arguments are malformed**
- Add format examples in parameter descriptions
- Use `enum` to restrict to valid values
- Validate arguments server-side (never trust LLM output blindly)

**Model calls wrong tool**
- Ensure tool names and descriptions are clearly distinct
- Reduce the total number of tools (< 10 for best reliability)
- Add explicit disambiguation in descriptions

---

## Key Takeaways

- Function calling outputs structured JSON tool calls — you execute the function, then return the result; the model never runs code directly
- Define tool schemas with precise `description` fields: what the tool does, when to use it, and what it returns
- Always execute all tool calls returned in a single response before making the next LLM call — parallel tool calls must all be resolved first
- Wrap every tool function in try/except and return descriptive error strings — tool errors that propagate cause confusing model behavior
- Use `enum` constraints in parameter schemas to restrict to valid values; use `required` to mark mandatory fields
- `tool_choice="required"` forces the model to call a tool; useful for extraction tasks where you always need structured output
- Reduce the tool set per request to under 10-15 tools — more tools degrade selection accuracy even when the model supports hundreds
- Test tool selection explicitly: build query-to-expected-tool-name assertions before deploying

---

## FAQ

**What is the difference between function calling and structured output?**
Function calling: the model decides to call a tool when it needs external data or capabilities, emitting a structured tool call object. Structured output (via `response_format`): force the model to return its final text answer as a specific JSON schema. They are complementary — you can use function calling for tool execution and structured output for the final response format.

**Can I use function calling with open-source models?**
Yes. Llama 3.1, Mistral, and Qwen2.5 support function calling. Use Ollama or vLLM with the `tools` parameter — the API format is compatible with OpenAI. Smaller models (under 7B) tend to be unreliable with complex schemas or multi-step tool chains; test with your specific model before assuming compatibility.

**How do I force the model to always call a specific tool?**
Set `tool_choice={"type": "function", "function": {"name": "your_tool_name"}}` to force a specific tool call. Use `tool_choice="required"` to force any tool call. This is useful for structured extraction tasks where you want the model to always fill a specific schema rather than answering from training data.

**What happens if the model produces invalid JSON in tool arguments?**
The API returns a tool call with potentially malformed arguments. Always parse and validate arguments before calling your function. Use `json.loads()` with a try/except wrapper. For LangChain agents, `handle_parsing_errors=True` returns the parse error as an observation rather than crashing.

**How does parallel tool calling work?**
When the model determines it needs multiple tools that can run concurrently, it returns multiple tool calls in a single response. You must execute all of them, collect all results, and return all results in a single follow-up message (one `tool` message per tool call ID). Only then do you make the next LLM call. Executing them sequentially and sending partial results breaks the protocol.

**What is the difference between OpenAI function calling and Anthropic tool use?**
The concepts are identical. OpenAI uses `tools` with `type: "function"` objects and returns `tool_calls` in the assistant message. Anthropic uses `tools` with `input_schema` and returns `tool_use` blocks in the content array. The execution loop is the same: execute tool calls, return results with matching IDs. LangChain and LiteLLM abstract both behind identical interfaces.

**How many tokens do tool schemas consume?**
Each tool schema (name + description + parameter descriptions) typically consumes 50-200 tokens. With 10 tools and 100 tokens each, that is 1,000 tokens added to every request. For cost-sensitive applications, dynamically select which tools to include based on the query rather than sending all tools on every call.

---

## What to Learn Next

- [LLM Function Calling: Build Tool-Using AI Apps](/blog/llm-function-calling/)
- [AI Agent Tool Use: APIs, Search, and Code Execution](/blog/agent-tools/)
- [AI Agent Fundamentals: How LLM Agents Think, Plan, and Act](/blog/ai-agent-fundamentals/)
- [Build AI Agents Step-by-Step](/blog/build-ai-agents/)
- [LangChain Agents: Build Tool-Using LLMs](/blog/langchain-agents/)
