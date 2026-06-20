"""Unit tests for ErrorExtractor: detection, parsing, clustering, frequency."""

from pylogtracer.preprocessing.smart_reader import get_file_content
from pylogtracer.preprocessing.error_extractor import ErrorExtractor
from pylogtracer.preprocessing.error_type_classifier import ErrorTypeClassifier


def _extract(sample_log_path):
    reader = get_file_content()
    logs = reader.fetch_logs_by_date(sample_log_path)["logs"]
    # Mirror LogTracer's offline path: a classifier with no LLM factory still
    # applies the pattern matchers (connection->ConnectionError, etc.).
    classifier = ErrorTypeClassifier(factory=None)
    return ErrorExtractor(gap_seconds=60, classifier=classifier).extract(logs)


def test_extract_counts_only_errors(sample_log_path):
    result = _extract(sample_log_path)
    # 3 error entries (2x INC1000001 + 1x INC1000002); INFO lines excluded.
    assert result["total_errors"] == 3


def test_extract_empty_input_is_safe():
    result = ErrorExtractor().extract([])
    assert result["total_errors"] == 0
    assert result["clusters"] == []
    assert result["last_cluster"] is None


def test_clustering_splits_on_time_gap(sample_log_path):
    result = _extract(sample_log_path)
    # 10:00:05 & 10:00:07 cluster together; 10:05:30 (>60s later) is separate.
    assert len(result["clusters"]) == 2
    # last_cluster is the most recent incident (the ValueError).
    assert result["last_cluster"][0]["error_type"] == "ValueError"


def test_frequency_is_sorted_desc(sample_log_path):
    result = _extract(sample_log_path)
    freq = result["frequency"]
    # "Database connection refused" and "Reconnect failed" both map to
    # ConnectionError via pattern matching -> count of 2, the most frequent.
    assert freq["ConnectionError"] == 2
    assert list(freq)[0] == "ConnectionError"


# ── Workstream C: level-aware error detection ──────────────────────
_LEVEL_LOG = [
    "2024-03-01 10:00:00 INFO  recovered from failed attempt cleanly",  # "failed" but INFO
    "2024-03-01 10:00:05 INFO  GET /api/error-rates 200 ok",            # "error" but INFO
    "2024-03-01 10:00:10 ERROR database is down",                       # real error
    "2024-03-01 10:00:15 WARNING disk almost full",                     # warning
]


def test_default_substring_mode_overcounts():
    # Default behavior: substring scan flags the INFO lines too (3 of 4).
    result = ErrorExtractor().extract(_LEVEL_LOG)
    assert result["total_errors"] == 3


def test_level_aware_counts_only_real_errors():
    result = ErrorExtractor(level_aware=True).extract(_LEVEL_LOG)
    assert result["total_errors"] == 1   # only the ERROR line


def test_level_aware_include_warnings():
    result = ErrorExtractor(level_aware=True, include_warnings=True).extract(_LEVEL_LOG)
    assert result["total_errors"] == 2   # ERROR + WARNING


def test_level_aware_falls_back_when_no_level_token():
    # A line with no recognizable level token still uses the substring scan.
    entries = ["2024-03-01 10:00:00 ConnectionError: boom"]   # no level word
    result = ErrorExtractor(level_aware=True).extract(entries)
    assert result["total_errors"] == 1
