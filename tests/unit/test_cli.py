"""Workstream E: CLI smoke tests (offline — no LLM actions)."""

import json

import pytest

from pylogtracer.cli import main, build_parser


def test_summary_pretty(sample_log_path, capsys):
    rc = main([sample_log_path, "--summary"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "summary" in out
    assert "total_errors" in out


def test_summary_json_is_parseable(sample_log_path, capsys):
    rc = main([sample_log_path, "--summary", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["summary"]["total_errors"] == 3


def test_search_action(sample_log_path, capsys):
    rc = main([sample_log_path, "--search", "INC1000001", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert json.loads(out)["search"]["total_found"] == 3


def test_duration_of_action(sample_log_path, capsys):
    rc = main([sample_log_path, "--duration-of", "INC1000001", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)["keyword_duration"]
    assert payload["found"] is True
    assert payload["duration_seconds"] == 25


def test_bare_invocation_defaults_to_summary(sample_log_path, capsys):
    rc = main([sample_log_path])
    assert rc == 0
    assert "summary" in capsys.readouterr().out


def test_cli_report_markdown(sample_log_path, capsys):
    rc = main([sample_log_path, "--report", "markdown"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "# Log Analysis Report" in out


def test_cli_report_html_to_file(sample_log_path, tmp_path, capsys):
    out = tmp_path / "r.html"
    rc = main([sample_log_path, "--report", "html", "--output", str(out)])
    assert rc == 0
    assert out.exists()
    assert "wrote html report" in capsys.readouterr().out


def test_missing_file_returns_error(capsys):
    rc = main(["does_not_exist_98765.log", "--summary"])
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as ei:
        build_parser().parse_args(["x.log", "--version"])
    assert ei.value.code == 0
