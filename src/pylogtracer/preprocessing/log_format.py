"""
log_format.py
=============
Single source of truth for log *format* concerns, kept out of smart_reader so
the same rules are reused everywhere (no more duplicated timestamp regexes):

  - canonical timestamp patterns + `extract_timestamp` / `has_timestamp`
  - `open_log`  : transparent gzip support (.gz) on top of plain files
  - `read_lines`: bounded/tail reads (max_lines / max_bytes) + rotated-file glob
  - `normalize_line`: flatten a JSON-lines record into the canonical
    "<timestamp> <LEVEL> <message>" text shape the rest of the pipeline speaks,
    so structured logs work with zero downstream changes.
"""

import os
import re
import gzip
import glob as _glob
import json
import logging
from collections import deque
from datetime import datetime
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

# Canonical timestamp formats — the ONE place these live.
TS_PATTERNS = [
    (r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "%Y-%m-%d %H:%M:%S"),
    (r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}", "%d-%m-%Y %H:%M:%S"),
    (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "%Y-%m-%dT%H:%M:%S"),
    (r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}", "%Y/%m/%d %H:%M:%S"),
]
_COMPILED = [(re.compile(p), f) for p, f in TS_PATTERNS]

# Default JSON field names probed when flattening structured logs.
_DEFAULT_JSON_KEYS = {
    "timestamp": ["timestamp", "time", "ts", "asctime", "@timestamp", "datetime"],
    "level": ["level", "levelname", "severity", "lvl", "loglevel"],
    "message": ["message", "msg", "event", "log", "text"],
}


# ── timestamps ─────────────────────────────────────────────────────
def extract_timestamp(line: str) -> Optional[datetime]:
    """Return the first parseable timestamp in `line`, or None."""
    for rx, fmt in _COMPILED:
        m = rx.search(line)
        if m:
            try:
                return datetime.strptime(m.group(), fmt)
            except ValueError:
                continue
    return None


def has_timestamp(line: str) -> bool:
    return extract_timestamp(line) is not None


# ── file opening + bounded reads ───────────────────────────────────
def open_log(path: str, encoding: str = "utf-8", errors: str = "replace"):
    """Open a log file, transparently handling gzip (.gz)."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding=encoding, errors=errors)
    return open(path, "r", encoding=encoding, errors=errors)


def gather_rotated(path: str) -> List[str]:
    """Return `path` plus rotated siblings (app.log.1, app.log.2.gz), oldest first."""
    matches = set(_glob.glob(path + "*"))
    if os.path.exists(path):
        matches.add(path)
    if not matches:
        return [path]
    return sorted(matches, key=lambda p: os.path.getmtime(p))


def _read_one(path: str, max_lines: Optional[int], max_bytes: Optional[int]) -> List[str]:
    """Read a single file with optional line/byte caps (most-recent window)."""
    is_gz = str(path).endswith(".gz")
    # Byte-bounded tail only makes sense on a seekable plain file.
    if max_bytes and not is_gz and not max_lines:
        size = os.path.getsize(path)
        with open(path, "rb") as fb:
            if size > max_bytes:
                fb.seek(size - max_bytes)
                fb.readline()  # discard the partial first line
            raw = fb.read().decode("utf-8", errors="replace")
        return raw.splitlines()
    with open_log(path) as f:
        if max_lines:
            return [ln.rstrip("\n") for ln in deque(f, maxlen=max_lines)]
        return [ln.rstrip("\n") for ln in f]


def read_lines(
    path: str,
    max_lines: Optional[int] = None,
    max_bytes: Optional[int] = None,
    log_format: str = "auto",
    json_keys: Optional[Dict] = None,
    glob_rotated: bool = False,
    log_pattern: Optional[str] = None,
    timestamp_format: Optional[str] = None,
) -> List[str]:
    """
    Read log lines with optional bounding + format normalization.

    Defaults (no caps, log_format="auto") reproduce a plain full read; "auto"
    only rewrites lines that are valid JSON objects, so plain text is untouched.

    A custom `log_pattern` (regex with named groups timestamp/level/message)
    takes precedence: matching lines are normalized into the canonical
    "<timestamp> <LEVEL> <message>" shape; non-matching lines pass through.
    """
    paths = gather_rotated(path) if glob_rotated else [path]
    lines: List[str] = []
    for p in paths:
        lines.extend(_read_one(p, max_lines, max_bytes))
    # If a global line cap is set across rotated files, keep the most recent.
    if glob_rotated and max_lines and len(lines) > max_lines:
        lines = lines[-max_lines:]

    if log_pattern:
        compiled = re.compile(log_pattern)
        lines = [apply_custom_pattern(ln, compiled, timestamp_format) for ln in lines]
    elif log_format != "text":
        lines = [normalize_line(ln, json_keys, mode=log_format) for ln in lines]
    return lines


def _normalize_custom_ts(ts: str, fmt: Optional[str]) -> str:
    """Normalize a captured timestamp using the user's strptime fmt, else best-effort."""
    if not ts:
        return ""
    if fmt:
        try:
            return datetime.strptime(ts.strip(), fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return _normalize_ts(ts)


def apply_custom_pattern(raw: str, compiled, timestamp_format: Optional[str] = None) -> str:
    """
    Apply a user-defined log-format regex (named groups: timestamp/level/message).

    On match, emit the canonical "<ts> <LEVEL> <message>" line the pipeline
    understands. On no match (blank lines, tracebacks), return the line as-is.
    """
    m = compiled.search(raw)
    if not m:
        return raw
    gd = m.groupdict()
    ts = _normalize_custom_ts(gd.get("timestamp") or "", timestamp_format)
    level = (gd.get("level") or "").upper()
    msg = gd.get("message")
    if msg is None:
        msg = raw  # no message group — keep the original text
    parts = [p for p in (ts, level, msg) if p]
    return " ".join(parts)


# ── JSON-lines normalization ───────────────────────────────────────
def _normalize_ts(ts: str) -> str:
    """Best-effort convert a JSON timestamp into canonical '%Y-%m-%d %H:%M:%S'."""
    if not ts:
        return ""
    dt = extract_timestamp(ts)
    if dt:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    clean = ts.strip().rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(clean, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    try:  # epoch seconds
        val = float(ts)
        if val > 1e8:
            return datetime.fromtimestamp(val).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OverflowError, OSError):
        pass
    return ts


def normalize_line(raw: str, json_keys: Optional[Dict] = None, mode: str = "auto") -> str:
    """
    Flatten a JSON-lines record to "<ts> <LEVEL> <message>".

    mode="text" → never touch the line. mode="auto" → only lines that look like
    a JSON object AND parse to a dict. mode="json" → try every line.
    Non-JSON / unparseable lines are returned unchanged.
    """
    if mode == "text":
        return raw
    s = raw.strip()
    if mode == "auto" and not s.startswith("{"):
        return raw
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return raw
    if not isinstance(obj, dict):
        return raw

    keys = json_keys or _DEFAULT_JSON_KEYS

    def pick(names):
        for n in names:
            if n in obj and obj[n] not in (None, ""):
                return str(obj[n])
        return ""

    ts = _normalize_ts(pick(keys.get("timestamp", [])))
    level = pick(keys.get("level", [])).upper()
    msg = pick(keys.get("message", []))
    if not msg:  # no message field — serialize remaining fields
        used = set(sum(keys.values(), []))
        msg = " ".join(f"{k}={v}" for k, v in obj.items() if k not in used)
    parts = [p for p in (ts, level, msg) if p]
    return " ".join(parts)
