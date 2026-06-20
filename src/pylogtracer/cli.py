"""
cli.py
======
Command-line interface for pylogtracer. Registered as the `pylogtracer`
console script (see pyproject.toml [project.scripts]).

Examples
--------
  pylogtracer app.log --summary
  pylogtracer app.log --frequency --since "10am"
  pylogtracer app.log --search INC5000002
  pylogtracer app.log --tail --max-lines 100000 --level-aware --health
  pylogtracer app.log --format json --health
  pylogtracer app.log --ask "what caused the crash?" \\
      --provider ollama --model qwen2.5:3b
  pylogtracer app.log --summary --json     # machine-readable output
"""

import sys
import json
import argparse
from typing import Optional, List

from pylogtracer import __version__
from pylogtracer.logtracer import LogTracer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pylogtracer",
        description="Provider-agnostic log analysis: summarize, search and ask about logs.",
    )
    p.add_argument("logfile", help="Path to the log file (.log, .txt, .jsonl, .gz)")
    p.add_argument("--version", action="version", version=f"pylogtracer {__version__}")

    # Actions (choose one or more).
    act = p.add_argument_group("actions")
    act.add_argument("--summary", action="store_true", help="High-level overview")
    act.add_argument("--frequency", action="store_true", help="Error counts by type")
    act.add_argument("--health", action="store_true", help="Health status (OK/WARNING/CRITICAL)")
    act.add_argument("--incident", action="store_true", help="Show the last incident")
    act.add_argument("--duration", action="store_true", help="Duration of the last error burst")
    act.add_argument("--duration-of", metavar="KEYWORD",
                     help="Time from first to last occurrence of any keyword/id/path")
    act.add_argument("--search", metavar="KEYWORD", help="Find log lines containing KEYWORD")
    act.add_argument("--related", metavar="ID", help="Lines + error cluster for an identifier")
    act.add_argument("--root-cause", action="store_true", help="LLM root-cause of last incident")
    act.add_argument("--ask", metavar="QUESTION", help="Ask a free-form question (LLM agent)")
    act.add_argument("--report", choices=["markdown", "md", "html"],
                     help="Generate a full report to stdout (or --output)")
    act.add_argument("--include-root-cause", action="store_true",
                     help="Include the LLM root-cause section in --report")
    act.add_argument("-o", "--output", metavar="PATH", help="Write --report to a file")

    # Time filtering.
    tf = p.add_argument_group("time filtering")
    tf.add_argument("--date", help="Filter to a date, e.g. 2026-06-19")
    tf.add_argument("--from", dest="from_dt", help="Range start 'YYYY-MM-DD HH:MM:SS'")
    tf.add_argument("--to", dest="to_dt", help="Range end 'YYYY-MM-DD HH:MM:SS'")

    # Read / format options.
    rd = p.add_argument_group("read / format")
    rd.add_argument("--tail", action="store_true", help="Read only a recent window")
    rd.add_argument("--max-lines", type=int, help="Read only the last N lines")
    rd.add_argument("--max-bytes", type=int, help="Read only the last N bytes")
    rd.add_argument("--format", dest="log_format", default="auto",
                    choices=["auto", "text", "json"], help="Log format (default auto)")
    rd.add_argument("--glob-rotated", action="store_true",
                    help="Also read rotated siblings (app.log.1, app.log.2.gz)")
    rd.add_argument("--level-aware", action="store_true",
                    help="Detect errors from the LEVEL field, not substrings")
    rd.add_argument("--include-warnings", action="store_true",
                    help="With --level-aware, also count WARN/WARNING")
    rd.add_argument("--log-pattern", metavar="REGEX",
                    help="Custom log regex with named groups (?P<timestamp>)(?P<level>)(?P<message>)")
    rd.add_argument("--timestamp-format", metavar="FMT",
                    help="strptime format for the captured timestamp group")

    # LLM options.
    llm = p.add_argument_group("llm (for --ask / --root-cause)")
    llm.add_argument("--provider", choices=["ollama", "openai", "anthropic", "custom"])
    llm.add_argument("--model")
    llm.add_argument("--base-url")
    llm.add_argument("--api-key")
    llm.add_argument("--cache", metavar="PATH", help="Persist learned keywords to PATH")
    llm.add_argument("--redact", dest="redact", action="store_true", default=None,
                     help="Force PII/secret redaction on")
    llm.add_argument("--no-redact", dest="redact", action="store_false",
                     help="Force PII/secret redaction off")

    p.add_argument("--json", action="store_true", help="Emit raw JSON instead of pretty text")
    return p


def _llm_config(args) -> Optional[dict]:
    # An LLM is only configured when a provider is given; otherwise library
    # mode (offline). --ask / --root-cause without --provider will fall back to
    # the factory default provider, which may fail if nothing is reachable.
    if not args.provider:
        return None
    cfg = {"provider": args.provider}
    if args.model:
        cfg["model"] = args.model
    if args.base_url:
        cfg["base_url"] = args.base_url
    if args.api_key:
        cfg["api_key"] = args.api_key
    return cfg


def _emit(label: str, value, as_json: bool):
    if as_json:
        print(json.dumps({label: value}, indent=2, default=str, ensure_ascii=False))
        return
    print(f"\n=== {label} ===")
    if isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, default=str, ensure_ascii=False))
    else:
        print(value)


def main(argv: Optional[List[str]] = None) -> int:
    # Logs (and reports) can contain non-ASCII; force UTF-8 so printing never
    # crashes on a legacy Windows console (cp1252).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)

    tracer = LogTracer(
        args.logfile,
        llm_config=_llm_config(args),
        cache_path=args.cache,
        level_aware=args.level_aware,
        include_warnings=args.include_warnings,
        redact=args.redact,
        max_lines=args.max_lines,
        max_bytes=args.max_bytes,
        tail=args.tail,
        log_format=args.log_format,
        glob_rotated=args.glob_rotated,
        log_pattern=args.log_pattern,
        timestamp_format=args.timestamp_format,
    )

    tf = {"date": args.date, "from_dt": args.from_dt, "to_dt": args.to_dt}

    # --report produces a whole document, not a labeled section — handle first.
    if args.report:
        fmt = "html" if args.report == "html" else "markdown"
        try:
            rep = tracer.generate_report(
                fmt, date=args.date, from_dt=args.from_dt, to_dt=args.to_dt,
                include_root_cause=args.include_root_cause, output=args.output,
            )
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"wrote {fmt} report to {args.output}" if args.output else rep)
        return 0

    # (enabled?, label, callable). Time-filterable actions take `tf`.
    actions = [
        (args.summary, "summary", lambda: tracer.summary(**tf)),
        (args.frequency, "frequency", lambda: tracer.error_frequency(**tf)),
        (args.health, "health", tracer.health_check),
        (args.incident, "last_incident", lambda: tracer.last_incident(**tf)),
        (args.duration, "incident_duration", lambda: tracer.incident_duration(**tf)),
        (args.duration_of, "keyword_duration", lambda: tracer.keyword_duration(args.duration_of)),
        (args.search, "search", lambda: tracer.search(args.search)),
        (args.related, "related", lambda: tracer.get_related_logs(args.related)),
        (args.root_cause, "root_cause", lambda: tracer.root_cause_analysis(**tf)),
        (args.ask, "answer", lambda: tracer.ask(args.ask)),
    ]

    ran = False
    try:
        for enabled, label, fn in actions:
            if enabled:
                _emit(label, fn(), args.json)
                ran = True
        if not ran:
            # Default action so a bare invocation is still useful.
            _emit("summary", tracer.summary(), args.json)
    except Exception as e:  # surface a clean error, not a traceback
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
