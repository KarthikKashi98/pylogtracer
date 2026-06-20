"""Shared pytest fixtures for the pylogtracer test suite."""

import gzip
import json

import pytest

# A small, deterministic log used across the offline tests. It contains:
#   - INFO lines (must NOT be treated as errors)
#   - A multi-line traceback that must be grouped into one entry
#   - Two ConnectionError lines within 60s -> one cluster (after merge)
#   - A later ValueError -> a separate cluster
#   - Shared incident id INC1000001 across ERROR + INFO lines
SAMPLE_LOG = """\
2024-03-01 09:59:00 INFO  Application started
2024-03-01 10:00:01 INFO  Handling request REQ-100
2024-03-01 10:00:05 ERROR Database connection refused - INC1000001
Traceback (most recent call last):
  File "app.py", line 42, in connect
    raise ConnectionError("timed out")
ConnectionError: timed out
2024-03-01 10:00:07 ERROR Reconnect failed - INC1000001
2024-03-01 10:00:30 INFO  Retrying connection - INC1000001
2024-03-01 10:05:30 ERROR Invalid value provided - INC1000002
ValueError: bad input
2024-03-01 11:30:00 INFO  Health ok
"""


@pytest.fixture
def sample_log_path(tmp_path):
    """Write SAMPLE_LOG to a temp file and return its path as a string."""
    p = tmp_path / "app.log"
    p.write_text(SAMPLE_LOG, encoding="utf-8")
    return str(p)


@pytest.fixture
def gzip_log_path(tmp_path):
    """SAMPLE_LOG written as a .gz file (same content, compressed)."""
    p = tmp_path / "app.log.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write(SAMPLE_LOG)
    return str(p)


@pytest.fixture
def jsonl_log_path(tmp_path):
    """A JSON-lines log: 1 INFO, 2 ERROR, 1 WARNING (one object per line)."""
    records = [
        {"timestamp": "2024-03-01 10:00:00", "level": "INFO", "message": "service started"},
        {"timestamp": "2024-03-01 10:00:05", "level": "ERROR", "message": "db connection refused"},
        {"timestamp": "2024-03-01 10:00:06", "level": "ERROR", "message": "retry attempt failed"},
        {"timestamp": "2024-03-01 11:00:00", "level": "WARNING", "message": "disk almost full"},
    ]
    p = tmp_path / "app.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return str(p)
