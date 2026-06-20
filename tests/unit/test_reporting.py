"""Feature 2: Markdown / HTML report generation."""

import pytest

from pylogtracer import LogTracer


def test_markdown_report_structure(sample_log_path):
    md = LogTracer(sample_log_path).generate_report("markdown")
    assert "# Log Analysis Report" in md
    assert "## Summary" in md
    assert "## Error frequency" in md
    assert "## Last incident" in md
    assert "ConnectionError" in md          # a real type from the sample
    assert md.endswith("\n")


def test_html_report_is_valid_html(sample_log_path):
    html = LogTracer(sample_log_path).generate_report("html")
    assert html.lstrip().lower().startswith("<!doctype html")
    assert "<table" in html
    assert "Log Analysis Report" in html
    assert html.rstrip().endswith("</html>")


def test_report_writes_file_and_returns_same(sample_log_path, tmp_path):
    out = tmp_path / "report.html"
    rep = LogTracer(sample_log_path).generate_report("html", output=str(out))
    assert out.exists()
    assert out.read_text(encoding="utf-8") == rep


def test_report_md_alias(sample_log_path):
    assert "# Log Analysis Report" in LogTracer(sample_log_path).generate_report("md")


def test_report_invalid_format_raises(sample_log_path):
    with pytest.raises(ValueError):
        LogTracer(sample_log_path).generate_report("pdf")


def test_html_report_escapes_content(tmp_path):
    # A message with HTML-special chars must be escaped in the HTML report.
    p = tmp_path / "x.log"
    p.write_text("2024-03-01 10:00:00 ERROR bad <script> & value\n", encoding="utf-8")
    html = LogTracer(str(p)).generate_report("html")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
