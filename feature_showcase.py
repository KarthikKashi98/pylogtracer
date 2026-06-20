"""
feature_showcase.py
===================
A single, self-documenting tour of EVERYTHING pylogtracer can do right now.

It runs against `sample_app.log` (dated 2026-06-19) and prints each feature
with its method signature and a one-line description, so you can see the full
public API and what each call returns.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
  # Library mode only — no LLM needed, fully offline, always works:
  python feature_showcase.py

  # Add the LLM-powered features (root cause + ask) — needs Ollama running:
  python feature_showcase.py --agent

  # Use a different model/provider for the --agent part:
  python feature_showcase.py --agent --model qwen2.5:7b
------------------------------------------------------------------------------

PUBLIC API AT A GLANCE
------------------------------------------------------------------------------
  Library mode (no LLM):
    summary(date?, from_dt?, to_dt?)        high-level overview
    error_frequency(date?, from_dt?, to_dt?) count errors by type
    errors_by_date(date)                    all errors on a date
    errors_in_range(from_dt, to_dt)         all errors between two timestamps
    last_incident()                         most recent error cluster
    incident_duration(...)                  how long the last incident lasted
    health_check()                          OK / WARNING / CRITICAL
    search(keyword, max_results)            find any log line (all levels)
    get_related_logs(identifier)            raw lines + matching error cluster
    get_entry_details(identifier)           full parsed entry + traceback

  Agent mode (needs llm_config):
    root_cause_analysis(...)                LLM root-cause of last incident
    ask("free form question")               LangGraph ReAct agent over the tools
------------------------------------------------------------------------------
"""

import sys
import json
import logging

from pylogtracer import LogTracer

LOG_FILE = "sample_app.log"
THE_DATE = "2026-06-19"          # the date used inside sample_app.log


# ── pretty printing helpers ───────────────────────────────────────
def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show(call_label, value, desc=""):
    """Print a labelled result, JSON-formatting dicts/lists, truncating noise."""
    print(f"\n>>> {call_label}")
    if desc:
        print(f"    ({desc})")
    if isinstance(value, (dict, list)):
        text = json.dumps(value, indent=2, default=str, ensure_ascii=False)
        if len(text) > 1500:
            text = text[:1500] + "\n    ... (truncated)"
        print(text)
    else:
        print(value)


def safe(fn):
    """Run a showcase step but never let one failure abort the whole tour."""
    try:
        fn()
    except Exception as e:  # pragma: no cover - demo resilience
        print(f"    [step failed: {type(e).__name__}: {e}]")


# ── LIBRARY MODE (no LLM) ──────────────────────────────────────────
def library_mode():
    t = LogTracer(LOG_FILE)   # no llm_config -> offline, regex/pattern only

    banner("1. OVERVIEW")
    safe(lambda: show("summary()", t.summary(),
                      "totals, error types, first/last error, frequency"))
    safe(lambda: show("error_frequency()", t.error_frequency(),
                      "count of each error type, most frequent first"))
    safe(lambda: show("health_check()", t.health_check(),
                      "is the system healthy? OK / WARNING / CRITICAL"))

    banner("2. TIME FILTERING")
    safe(lambda: show(f'errors_by_date("{THE_DATE}")',
                      _slim(t.errors_by_date(THE_DATE)),
                      "every error on a given day"))
    safe(lambda: show(f'errors_in_range("{THE_DATE} 10:00:00", "{THE_DATE} 12:00:00")',
                      _slim(t.errors_in_range(f"{THE_DATE} 10:00:00", f"{THE_DATE} 12:00:00")),
                      "errors inside a time window"))
    safe(lambda: show(f'error_frequency(from_dt="{THE_DATE} 09:00:00", to_dt="{THE_DATE} 15:00:00")',
                      t.error_frequency(from_dt=f"{THE_DATE} 09:00:00", to_dt=f"{THE_DATE} 15:00:00"),
                      "frequency, but only within a window"))

    banner("3. INCIDENTS")
    safe(lambda: show("last_incident()", _slim(t.last_incident()),
                      "the most recent error cluster"))
    safe(lambda: show("incident_duration()", t.incident_duration(),
                      "start/end/duration of the last incident"))

    banner("4. SEARCH & LOOKUP")
    safe(lambda: show('search("INC5000002")', _slim_search(t.search("INC5000002")),
                      "find ANY log line (INFO/WARN/ERROR/CRITICAL) by keyword/id"))
    safe(lambda: show('get_related_logs("INC5000003")', t.get_related_logs("INC5000003"),
                      "raw matching lines PLUS the matching error cluster"))
    safe(lambda: show('get_entry_details("INC5000001")', t.get_entry_details("INC5000001"),
                      "full parsed entry incl. traceback for an identifier"))


# ── AGENT / LLM MODE ───────────────────────────────────────────────
def agent_mode(model):
    cfg = {"provider": "ollama", "model": model, "base_url": "http://localhost:11434"}
    t = LogTracer(LOG_FILE, llm_config=cfg)

    banner(f"5. LLM CLASSIFICATION  (model={model})")
    safe(lambda: show("error_frequency()  [LLM-assisted]", t.error_frequency(),
                      "unknown error types now classified by the LLM"))
    safe(lambda: show("classifier keyword store", t._classifier.get_keyword_store(),
                      "keywords the LLM learned this session (free reuse next time)"))

    banner("6. ROOT CAUSE ANALYSIS")
    safe(lambda: show("root_cause_analysis()", t.root_cause_analysis(),
                      "LLM analyses the last incident; may fetch more context on demand"))

    banner("7. ASK — NATURAL LANGUAGE (ReAct agent)")
    for q in [
        "is the system healthy?",
        "show all logs for INC5000005",
        "how many errors happened today and what types?",
    ]:
        safe(lambda q=q: show(f'ask("{q}")', t.ask(q)))


# ── trim helpers so the demo output stays readable ─────────────────
def _slim(errors):
    return [
        {"timestamp": str(e.get("timestamp")),
         "error_type": e.get("error_type"),
         "primary_error": (e.get("primary_error") or "")[:70]}
        for e in errors
    ]


def _slim_search(res):
    return {
        "keyword": res.get("keyword"),
        "total_found": res.get("total_found"),
        "entries": [line.splitlines()[0] for line in res.get("entries", [])],
    }


if __name__ == "__main__":
    # Show pylogtracer's own progress logs; silence the HTTP libraries.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    model = "qwen2.5:3b"
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]

    library_mode()

    if "--agent" in sys.argv:
        agent_mode(model)
    else:
        banner("LLM FEATURES SKIPPED")
        print("Add --agent to also run root_cause_analysis() and ask() via Ollama.")
        print("  python feature_showcase.py --agent")
