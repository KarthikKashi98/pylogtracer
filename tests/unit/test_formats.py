"""Workstream D: bounded/tail reads, gzip, and JSON-lines format support."""

from pylogtracer import LogTracer
from pylogtracer.preprocessing.smart_reader import get_file_content


# ── bounded / tail reads ───────────────────────────────────────────
def test_max_lines_keeps_recent_window(tmp_path):
    p = tmp_path / "big.log"
    lines = [f"2024-03-01 10:00:{i:02d} INFO line number {i}" for i in range(30)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reader = get_file_content(max_lines=10)
    res = reader.fetch_logs_by_date(str(p))
    assert res["total_matched"] == 10          # only the last 10 timestamped lines
    assert res.get("truncated") is True
    assert "line number 29" in reader._all_lines[-1]
    assert "line number 20" in reader._all_lines[0]


def test_max_bytes_tail_drops_oldest(tmp_path):
    p = tmp_path / "big.log"
    p.write_text(
        "2024-03-01 10:00:00 INFO first line here\n"
        "2024-03-01 10:00:01 INFO second line here\n"
        "2024-03-01 10:00:02 INFO third line here\n",
        encoding="utf-8",
    )
    reader = get_file_content(max_bytes=45)
    res = reader.fetch_logs_by_date(str(p))
    assert res.get("truncated") is True
    joined = "\n".join(reader._all_lines)
    assert "third line here" in joined
    assert "first line here" not in joined     # oldest dropped by the byte cap


def test_no_cap_is_not_truncated(sample_log_path):
    res = get_file_content().fetch_logs_by_date(sample_log_path)
    assert "truncated" not in res              # default read is unbounded


# ── gzip ────────────────────────────────────────────────────────────
def test_gzip_reads_same_as_plain(gzip_log_path):
    res = get_file_content().fetch_logs_by_date(gzip_log_path)
    # Same 7 grouped entries as the plain-text SAMPLE_LOG fixture.
    assert res["total_matched"] == 7


# ── JSON-lines ──────────────────────────────────────────────────────
def test_jsonl_classified_via_level(jsonl_log_path):
    t = LogTracer(jsonl_log_path, log_format="json", level_aware=True)
    s = t.summary()
    assert s["total_errors"] == 2              # 2 ERROR lines (WARNING excluded)


def test_jsonl_include_warnings(jsonl_log_path):
    t = LogTracer(jsonl_log_path, log_format="json", level_aware=True, include_warnings=True)
    assert t.summary()["total_errors"] == 3    # 2 ERROR + 1 WARNING


def test_auto_mode_leaves_plain_text_untouched(sample_log_path):
    # log_format defaults to "auto"; plain text must behave exactly as before.
    t = LogTracer(sample_log_path)
    assert t.summary()["total_errors"] == 3


# ── Feature 3: custom regex parser ─────────────────────────────────
def test_custom_log_pattern(tmp_path):
    p = tmp_path / "custom.log"
    p.write_text(
        "21/06/2026-10:00:05 | ERROR | Database connection lost\n"
        "21/06/2026-10:00:06 | INFO | Reconnecting\n"
        "21/06/2026-10:00:07 | CRITICAL | Service halted\n",
        encoding="utf-8",
    )
    pattern = (r"(?P<timestamp>\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2})\s*\|\s*"
               r"(?P<level>\w+)\s*\|\s*(?P<message>.*)")
    t = LogTracer(str(p), log_pattern=pattern,
                  timestamp_format="%d/%m/%Y-%H:%M:%S", level_aware=True)
    s = t.summary()
    assert s["total_entries"] == 3
    assert s["total_errors"] == 2                       # ERROR + CRITICAL (INFO excluded)
    assert s["first_error"] == "2026-06-21 10:00:05"    # normalized to canonical format
    assert "Database" in t.search("Database")["entries"][0]
