"""Unit tests for LogTracer library-mode methods (no LLM required)."""

from pylogtracer import LogTracer


def test_summary_offline(sample_log_path):
    t = LogTracer(sample_log_path)
    s = t.summary()
    assert s["total_errors"] == 3
    assert s["total_clusters"] == 2
    assert "ConnectionError" in s["error_types"]


def test_offline_classifier_has_no_factory(sample_log_path):
    # Plain LogTracer("app.log") must be fully offline: the classifier
    # should not be wired to an LLM factory.
    t = LogTracer(sample_log_path)
    assert t._llm_configured is False
    assert t._classifier.factory is None


def test_health_check_reports_warning(sample_log_path):
    t = LogTracer(sample_log_path)
    hc = t.health_check()
    assert hc["healthy"] is False
    assert hc["status"] in ("WARNING", "CRITICAL")
    assert hc["total_errors"] == 3


def test_get_related_logs_returns_cluster(sample_log_path):
    t = LogTracer(sample_log_path)
    grl = t.get_related_logs("INC1000001")

    # Full documented shape is present (regression: keys were missing).
    for key in (
        "all_entries", "error_cluster", "cluster_index",
        "total_in_cluster", "has_error_cluster", "note",
    ):
        assert key in grl

    assert grl["found"] is True
    assert grl["total_found"] == 3            # ERROR + ERROR + INFO lines
    assert grl["has_error_cluster"] is True
    assert grl["total_in_cluster"] == 2       # the two ConnectionError entries


def test_get_related_logs_not_found(sample_log_path):
    t = LogTracer(sample_log_path)
    grl = t.get_related_logs("NOPE-404")
    assert grl["found"] is False
    assert grl["total_found"] == 0
    assert grl["has_error_cluster"] is False


def test_incident_duration_shape(sample_log_path):
    t = LogTracer(sample_log_path)
    dur = t.incident_duration()
    assert "duration_seconds" in dur
    assert "duration_human" in dur
    assert dur["error_count"] >= 1


def test_keyword_duration_first_to_last(sample_log_path):
    # INC1000001 spans 10:00:05 (ERROR) .. 10:00:30 (INFO) = 25s, regardless of
    # whether those lines are "errors" or which cluster they belong to.
    t = LogTracer(sample_log_path)
    d = t.keyword_duration("INC1000001")
    assert d["found"] is True
    assert d["occurrences"] == 3
    assert d["first_occurrence"] == "2024-03-01 10:00:05"
    assert d["last_occurrence"] == "2024-03-01 10:00:30"
    assert d["duration_seconds"] == 25


def test_keyword_duration_works_for_non_error_keyword(sample_log_path):
    # A plain keyword that isn't an error/incident at all still gets a span.
    t = LogTracer(sample_log_path)
    d = t.keyword_duration("INFO")
    assert d["found"] is True
    assert d["duration_seconds"] >= 0


def test_keyword_duration_not_found(sample_log_path):
    t = LogTracer(sample_log_path)
    d = t.keyword_duration("NOPE-404")
    assert d["found"] is False
    assert d["occurrences"] == 0


def test_search_scoped_by_date(tmp_path):
    # Same keyword, different value per date — date scoping must isolate one.
    p = tmp_path / "pred.log"
    p.write_text(
        "2024-03-01 10:00:00 INFO prediction for MODEL-X = 0.85\n"
        "2024-03-02 10:00:00 INFO prediction for MODEL-X = 0.42\n"
        "2024-03-03 10:00:00 INFO prediction for MODEL-X = 0.13\n",
        encoding="utf-8",
    )
    t = LogTracer(str(p))
    assert t.search("MODEL-X")["total_found"] == 3            # unscoped: all dates
    scoped = t.search("MODEL-X", date="2024-03-01")
    assert scoped["total_found"] == 1
    assert "0.85" in scoped["entries"][0] and "0.42" not in scoped["entries"][0]


def test_search_scoped_by_range(tmp_path):
    p = tmp_path / "pred.log"
    p.write_text(
        "2024-03-01 09:00:00 INFO MODEL-X = 0.85\n"
        "2024-03-01 15:00:00 INFO MODEL-X = 0.42\n",
        encoding="utf-8",
    )
    t = LogTracer(str(p))
    res = t.search("MODEL-X", from_dt="2024-03-01 08:00:00", to_dt="2024-03-01 12:00:00")
    assert res["total_found"] == 1 and "0.85" in res["entries"][0]


def test_read_is_memoized_per_filter(sample_log_path, monkeypatch):
    # Count actual disk reads. summary() used to read TWICE (direct _read +
    # _get_extraction); memoization must collapse same-filter reads to one.
    import pylogtracer.preprocessing.log_format as lf
    calls = {"n": 0}
    real = lf.read_lines

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(lf, "read_lines", counting)
    t = LogTracer(sample_log_path)

    t.summary()
    assert calls["n"] == 1, "summary() should read the file only once"

    # Same-filter follow-ups reuse the cache — no further reads.
    t.summary()
    t.error_frequency()
    t.last_incident()
    assert calls["n"] == 1
