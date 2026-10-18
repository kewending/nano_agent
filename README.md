# 🤖 nano-agent

A lightweight, zero-framework AI agent built from scratch in Python 3.12 to demystify how **Agentic Applications**, **ReAct loops**, and **Tool Calling** actually work under the hood.

---

## 🎯 Purpose of This Project

High-level frameworks (like LangChain, CrewAI, or AutoGen) often hide the core mechanisms behind layers of complex abstractions. 

**`nano-agent`** is my personal educational learning project designed to strip away the "magic" and demonstrate that an autonomous AI agent is fundamentally:
1. **A Stateless Model Client**: Interacting through the standard OpenAI-compatible API format.
2. **A Memory Manager**: Appending and accumulating conversation context across turns.
3. **A Tool Schema Introspector**: Converting standard Python functions into JSON Schemas at runtime using Python's `inspect` module.
4. **An Autonomous Execution Loop (ReAct)**: Alternating between model reasoning, local tool execution, and context feedback until a task is resolved.

---

## 🧠 How the Agent Loop Works (ReAct)

When you ask the agent a question, it enters an autonomous **Reasoning + Acting** loop bounded by a maximum iteration limit:

```
                  ┌──────────────────────────────┐
                  │      User Question / Goal    │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                    ┌───────────────────────────┐
                    │  Append to Memory History │
                    └────────────┬──────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               │
      ┌─────────────────────┐                    │
      │ Call LLM with Tools │ <──────────────┐   │
      └──────────┬──────────┘                │   │
                 │                           │   │
        Is tool requested?                   │   │
        /                \                   │   │
      YES                 NO                 │   │
      /                     \                │   │
     ▼                       ▼               │   │
┌──────────────┐     ┌─────────────────────┐ │   │
│ Execute Tool │     │ Save Final Answer   │ │   │
│  in Python   │     │ & Return to User    │ │   │
└──────┬───────┘     └─────────────────────┘ │   │
       │                                     │   │
       ▼                                     │   │
┌──────────────┐                             │   │
│ Feed Result  ├─────────────────────────────┘   │
│ Back as Tool │                                 │
│   Message    │                                 │
└──────────────┘                                 │
                                                 ▼
                                     ┌───────────────────────┐
                                     │  Print Cumulative     │
                                     │     Token Usage       │
                                     └───────────────────────┘
```

---

## 🏗️ Architecture & Project Structure

```text
nano-agent/
├── pyproject.toml              # Project dependencies managed via uv
├── uv.lock                     # Deterministic dependency lockfile
├── README.md                   # Project overview & documentation
├── .env                        # Local API keys and endpoints (ignored by git)
└── src/
    └── nano_agent/
        ├── __init__.py         # Interactive CLI REPL entrypoint
        ├── config.py           # Multi-provider configuration (Local llama / Gemini)
        ├── types.py            # Token tracking (input, output, cached, thinking)
        ├── memory.py           # Multi-turn conversation state manager
        ├── tools.py            # Dynamic @tool decorator and schema generator
        ├── agent.py            # The autonomous ReAct execution loop
        └── prompts/
            └── system.md       # Isolated, version-controlled system prompt
```

---

## ✨ Key Features

* **Multi-Provider Support**: Seamlessly switch between local models (running via `llama-server` / `llama.cpp` on localhost) and cloud models (Google Gemini via its OpenAI-compatible endpoint) without rewriting code.
* **Automatic Tool Schema Generation**: Write normal Python functions with type hints and docstrings; the `@tool` decorator automatically builds the OpenAI function schema using `inspect`.
* **Granular Token Analytics**: Tracks prompt tokens, completion tokens, prompt cache hits, and hidden model reasoning/thinking tokens across both single turns and entire sessions.

---

## 🚀 Quickstart

### 1. Requirements
* [uv](https://docs.astral.sh/uv/) (Modern Python package manager)
* Python 3.12
* Cloud API or [llama.cpp](https://llama.app/)

* Optional [Hugging Face](https://huggingface.co/) account account for downloading various models to play with.

### 2. Configure Environment
Create a `.env` file in the root directory:

```ini
# Option A: Local llama.cpp server (e.g. Gemma 4)
LLAMA_API_KEY=not-needed
LLAMA_BASE_URL=http://localhost:9931/v1
LLAMA_MODEL=ggml-org/gemma-4-E4B-it-GGUF:Q8_0

# Option B: Google Gemini Cloud (via OpenAI-compatible endpoint)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL=gemini-3.8-flash

# Operational Limits
MAX_ITERATIONS=10
```

### 3. Run the Agent
Run the project using `uv`:

```powershell
uv run nano-agent
```

---

## 💬 Example Interactions

```text
You: What is the current time?
🔧 Model requested tool execution:
   ▶ get_current_time({})
   ✔ Output: 2026-10-18 09:31:02 AUS Eastern Daylight Time (+1100)
Assistant: The current time is 2026-10-18 09:31:02 AUS Eastern Daylight Time (+1100).

You: Can you multiply 123 by 456, then add 50?
🔧 Model requested tool execution:
   ▶ calculate({"expression":"123 * 456 + 50"})
   ✔ Output: 56138
Assistant: 123 * 456 + 50 equals 56,138.

You: quit
Goodbye!
--------------------------------------------------
📈 Total Session Usage: 📊 Tokens: 376 in (251 cached) | 330 out | 706 total
```

---

## 📚 What This Project Taught Me

1. **LLMs are purely stateless**: Multi-turn "memory" is simply resending the entire message history back to the model on every request.
2. **LLMs cannot run code directly**: When an LLM "calls a tool", it merely returns structured JSON asking the client to execute the function and report back the result.
3. **Safety first**: Tool executions must always catch exceptions gracefully and return error strings to the model so the agent can self-correct instead of crashing.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).