"""Unit tests for smart_reader: entry grouping and keyword search."""

from pylogtracer.preprocessing.smart_reader import get_file_content


def test_group_into_entries_groups_traceback(sample_log_path):
    reader = get_file_content()
    result = reader.fetch_logs_by_date(sample_log_path)

    entries = result["logs"]
    # The ERROR line plus its 4 traceback lines must collapse to ONE entry.
    db_entries = [e for e in entries if "Database connection refused" in e]
    assert len(db_entries) == 1
    assert "ConnectionError: timed out" in db_entries[0]
    assert "File \"app.py\"" in db_entries[0]


def test_fetch_no_filter_returns_all_entries(sample_log_path):
    reader = get_file_content()
    result = reader.fetch_logs_by_date(sample_log_path)
    assert result["filter"] is None
    # 7 timestamped lines -> 7 grouped entries (traceback folds into its line)
    assert result["total_matched"] == 7


def test_search_logs_is_recent_first_and_capped(sample_log_path):
    reader = get_file_content()
    reader.fetch_logs_by_date(sample_log_path)

    res = reader.search_logs("INC1000001")
    assert res["total_found"] == 3
    # Most-recent-first: the INFO "Retrying" line (10:00:30) precedes the
    # earlier ERROR lines.
    assert "Retrying connection" in res["entries"][0]

    capped = reader.search_logs("INC1000001", max_results=1)
    assert len(capped["entries"]) == 1


def test_search_before_load_returns_error():
    reader = get_file_content()
    res = reader.search_logs("anything")
    assert "error" in res


def test_fetch_missing_file_returns_error():
    reader = get_file_content()
    res = reader.fetch_logs_by_date("does_not_exist_12345.log")
    assert "error" in res
