# pylogtracer

> Provider-agnostic log analysis package with LLM support and a dynamic ReAct agent.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-pylogtracer-orange)](https://pypi.org/project/pylogtracer/)

---

## What is pylogtracer?

`pylogtracer` is a Python package that helps you analyze log files using LLMs. It works in two modes:

- **Library mode** — direct function calls, no agent needed, works without LLM
- **Agent mode** — ask free-form questions, a LangGraph ReAct agent decides which tools to call

It supports any LLM provider — Ollama (local), OpenAI, Anthropic, or any custom API.

---

## Features

- **Provider agnostic** — OpenAI, Anthropic, Ollama, or any OpenAI-compatible API (vLLM, llama.cpp, Groq)
- **Keyword learning** — LLM learns error patterns once, reuses them for free next time (persists across runs)
- **ReAct agent** — multi-step reasoning, calls multiple tools per question
- **Grounded answers** — counts, durations, log lines come straight from the data; the model never invents them, and says "not found" instead of guessing
- **Time resolver** — understands `"10am"`, `"yesterday"`, `"2 hours ago"`, `"between 9am and 11am"`
- **Date-scoped search** — find a keyword's value on a specific date (`search("MODEL-X", date="2024-03-01")`)
- **Cluster analysis** — groups related errors into incidents automatically
- **Reports** — generate Markdown/HTML summaries (`generate_report`)
- **Real logs** — gzip, JSON-lines, rotated files, custom formats, and huge files (bounded/tail reads)
- **No .env required** — pass config directly to `LogTracer`

---

## Installation

The core install is light (library mode works offline). Add an extra only for the
LLM provider you actually use:

```bash
pip install pylogtracer            # core: library mode, reports, custom parsers
pip install pylogtracer[ollama]    # + Ollama (local models)
pip install pylogtracer[openai]    # + OpenAI
pip install pylogtracer[anthropic] # + Anthropic
pip install pylogtracer[agent]     # + LangGraph ReAct agent (ask())
pip install pylogtracer[all]       # everything
```

> Agent mode (`ask()`) needs the `agent` extra; LLM classification / root-cause
> need a provider extra. Library-mode methods (`summary`, `search`,
> `error_frequency`, `generate_report`, …) work with just the core install.

---

## Quick Start

```python
from pylogtracer import LogTracer

# Ollama (local, no API key needed)
tracer = LogTracer(
    file_path  = "app.log",
    llm_config = {
        "provider": "ollama",
        "model":    "qwen2.5:7b",
        "base_url": "http://localhost:11434"
    }
)

# Library mode — no LLM needed
print(tracer.summary())
print(tracer.error_frequency())
print(tracer.health_check())

# Agent mode — LLM required
print(tracer.ask("what caused the crash at 10am?"))
print(tracer.ask("show INC1000004 related logs and how long it lasted"))
```

---

## Supported Providers

| Provider  | Example Model                   | API Key  |
|-----------|---------------------------------|----------|
| Ollama    | `qwen2.5:7b`, `llama3`, `mistral` | No     |
| OpenAI    | `gpt-4o-mini`, `gpt-4o`         | Yes      |
| Anthropic | `claude-3-5-haiku-20241022`     | Yes      |
| Custom    | Any OpenAI-compatible API       | Optional |

```python
# OpenAI
tracer = LogTracer("app.log", llm_config={
    "provider": "openai",
    "model":    "gpt-4o-mini",
    "api_key":  "sk-..."
})

# Anthropic
tracer = LogTracer("app.log", llm_config={
    "provider": "anthropic",
    "model":    "claude-3-5-haiku-20241022",
    "api_key":  "sk-ant-..."
})

# Custom / vLLM / LM Studio
tracer = LogTracer("app.log", llm_config={
    "provider": "custom",
    "model":    "my-model",
    "base_url": "http://my-server:8000/v1",
    "api_key":  "optional"
})
```

---

## Library Mode — All Methods

```python
tracer = LogTracer("app.log")   # no LLM needed for library mode

# Overview
tracer.summary()
# {'total_entries': 100, 'total_errors': 30, 'total_clusters': 11,
#  'error_types': [...], 'first_error': '...', 'last_error': '...'}

# Error counts
tracer.error_frequency()
tracer.error_frequency(date="2024-03-01")
tracer.error_frequency(from_dt="2024-03-01 09:00:00", to_dt="2024-03-01 11:00:00")

# Filter errors
tracer.errors_by_date("2024-03-01")
tracer.errors_in_range("2024-03-01 09:00:00", "2024-03-01 11:00:00")

# Last incident
tracer.last_incident()

# System health
tracer.health_check()
# {'healthy': False, 'status': 'CRITICAL', 'total_errors': 30, ...}

# Incident duration
tracer.incident_duration()
# {'start': '...', 'end': '...', 'duration_human': '6 minutes 12 seconds', ...}

# Search (any keyword / id / snippet)
tracer.search("INC1000001")              # by incident ID
tracer.search("connection refused")      # by keyword
tracer.search("MODEL-X", date="2024-03-01")  # scope to one date (same key, per-date value)
tracer.get_related_logs("INC1000004")   # all logs in same cluster
tracer.get_entry_details("INC1000004")  # full entry with traceback

# Duration of ANY keyword/id (first -> last occurrence)
tracer.keyword_duration("INC1000001")
tracer.incident_duration()               # the most recent error burst

# Reports (no LLM needed)
tracer.generate_report("markdown")
tracer.generate_report("html", output="report.html")

# Root cause (LLM required)
tracer.root_cause_analysis()
```

---

## Agent Mode — Ask Anything

```python
tracer = LogTracer("app.log", llm_config={...})

# Simple questions
tracer.ask("what is the last error?")
tracer.ask("is the system healthy?")
tracer.ask("how many DB errors happened?")

# Time-based (auto-resolved — no need to specify exact timestamps)
tracer.ask("what errors happened at 10am?")
tracer.ask("show errors from yesterday")
tracer.ask("what happened 2 hours ago?")
tracer.ask("errors between 9am and 11am")

# Identifier search
tracer.ask("show me INC1000004 related logs")
tracer.ask("what happened with REQ-456?")

# Date-scoped value lookup (same key can differ per date)
tracer.ask("what was the prediction for MODEL-X on 2024-03-01?")
tracer.ask("how long did INC1000002 last?")

# Multi-step (agent calls multiple tools automatically)
tracer.ask("what caused the crash and how long did it last?")
tracer.ask("compare errors today vs yesterday")
tracer.ask("show INC1000004 related logs and diagnose the root cause")
```

---

## How the Agent Works

The agent uses a **LangGraph ReAct loop** — it thinks, calls a tool, sees the result, and decides whether to call another tool or answer:

```
User: "what caused the crash and how long did it last?"
        ↓
  [think] → I need last_incident first
        ↓
  [tool]  → last_incident() → sees cluster
        ↓
  [think] → now I need root_cause and duration
        ↓
  [tool]  → root_cause() → LLM analysis
        ↓
  [tool]  → incident_duration() → 6 minutes 12 seconds
        ↓
  [think] → I have everything now
        ↓
  FINAL_ANSWER: "The crash was caused by..."
```

---

## Time Resolution

The agent automatically understands relative time — no need for exact timestamps:

| You say              | Resolved to                        |
|----------------------|------------------------------------|
| `"10am"`             | today 10:00:00 → 10:59:59          |
| `"yesterday 2pm"`    | yesterday 14:00:00 → 14:59:59      |
| `"this morning"`     | today 06:00:00 → 12:00:00          |
| `"2 hours ago"`      | now - 2h → now                     |
| `"last 30 minutes"`  | now - 30m → now                    |
| `"last night"`       | yesterday 20:00:00 → today 06:00:00|
| `"March 1"`          | 2024-03-01                         |

---

## Architecture

```
from pylogtracer import LogTracer      ← single entry point

LogTracer
    ├── preprocessing/
    │   ├── smart_reader.py            log reading, filtering, search
    │   ├── error_extractor.py         clustering, deduplication
    │   └── error_type_classifier.py   regex + keyword learning + LLM
    │
    ├── agents/
    │   ├── qa_agent.py                LangGraph ReAct agent (ask())
    │   └── root_cause_analyzer.py     LLM root cause analysis
    │
    ├── multiagent/
    │   └── context_bridge.py          agent-to-agent context loop
    │
    ├── llm/
    │   └── llm_factory.py             provider-agnostic LLM factory
    │
    └── utils/
        └── time_resolver.py           relative time resolution
```

---

## How Keyword Learning Works

The classifier uses a 3-pass system to minimize LLM calls:

```
Pass 1 — Named exception regex (free):
  "ConnectionError: timed out"  → ConnectionError ✓

Pass 2 — Keyword store (free, learned this session):
  "database connection refused" → DatabaseConnectionError ✓
  (learned from a previous LLM call this session)

Pass 3 — LLM batch (only truly unknown errors):
  LLM classifies + returns keywords for future use
  Keywords stored → next similar error is FREE
```

---

## Configuration Options

```python
LogTracer(
    file_path   = "app.log",    # path to log file (.log/.txt/.jsonl/.gz)
    llm_config  = {             # LLM provider config (None = library mode)
        "provider":    "ollama",
        "model":       "qwen2.5:7b",
        "base_url":    "http://localhost:11434",
        "api_key":     "optional",
        "temperature": 0.0,
        "max_tokens":  1024,
    },
    gap_seconds = 60,           # seconds between entries to split incidents
    max_retries = 2,            # max times LLM can request more context

    # ── 0.2.0 — robustness & cost ──────────────────────────────────
    cache_path  = ".plt_cache.json",  # persist learned keywords across runs
    max_context_tokens = None,        # override model context window for batching
    level_aware = False,        # detect errors from the LEVEL field, not substrings
    include_warnings = False,   # with level_aware, also count WARN/WARNING

    # ── 0.2.0 — large files & formats ─────────────────────────────
    tail        = False,        # read only a recent window (huge logs)
    max_lines   = None,         # read only the last N lines
    max_bytes   = None,         # read only the last N bytes
    log_format  = "auto",       # "auto" | "text" | "json" (JSON-lines)
    json_keys   = None,         # override JSON timestamp/level/message keys
    glob_rotated = False,       # also read app.log.1, app.log.2.gz

    # ── 0.2.0 — trust ─────────────────────────────────────────────
    redact      = None,         # None=auto (on for cloud, off for local Ollama)
    evidence    = True,         # ask() answers carry the supporting log lines

    # ── custom log format (any layout) ────────────────────────────
    log_pattern = None,         # regex w/ named groups (timestamp/level/message)
    timestamp_format = None,    # strptime fmt for the captured timestamp
)
```

> **Cost note:** with `cache_path` set, error types the LLM classified in a
> previous run are recognized for free, so the tokens sent to the model stay
> roughly flat no matter how large the log grows.

---

## Command-Line Interface

Installing the package also installs a `pylogtracer` command:

```bash
pylogtracer app.log --summary
pylogtracer app.log --frequency --health
pylogtracer app.log --search INC5000002
pylogtracer app.log --tail --max-lines 100000 --level-aware --health
pylogtracer app.log --format json --health         # JSON-lines logs
pylogtracer app.log --summary --json                # machine-readable output
pylogtracer app.log --report markdown               # full Markdown report
pylogtracer app.log --report html -o report.html    # HTML report to a file

# Agent mode (LLM):
pylogtracer app.log --ask "what caused the crash?" \
    --provider ollama --model qwen2.5:3b
```

---

## Reports

Generate a shareable report (no LLM needed):

```python
print(tracer.generate_report("markdown"))
tracer.generate_report("html", output="report.html")
tracer.generate_report("markdown", include_root_cause=True)  # adds LLM root cause
```

---

## Custom log formats

Point pylogtracer at *any* layout with a regex (named groups
`timestamp` / `level` / `message`); matching lines are normalized internally so
every feature still works:

```python
tracer = LogTracer(
    "weird.log",
    log_pattern = r"(?P<timestamp>\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2})\s*\|\s*"
                  r"(?P<level>\w+)\s*\|\s*(?P<message>.*)",
    timestamp_format = "%d/%m/%Y-%H:%M:%S",
    level_aware = True,
)
```

Built-in formats (`YYYY-MM-DD HH:MM:SS`, ISO `T`, `DD-MM-YYYY`, `YYYY/MM/DD`),
JSON-lines, and gzip are detected automatically — a custom pattern is only for
non-standard layouts.

---

## Requirements

```
langchain>=0.2.0
langchain-core>=0.2.0
langchain-openai>=0.1.0
langchain-anthropic>=0.1.0
langchain-ollama>=0.1.0
langgraph>=0.1.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

---

## Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_smart_reader.py -v
```

---

## CI/CD

Every push to `main` runs tests on Python 3.10, 3.11, and 3.12.
Every GitHub Release automatically publishes to PyPI.

---

## License

MIT

---

## Contributing

Pull requests welcome! Please run tests before submitting.

```bash
pip install -e .
pytest tests/ -v
```
