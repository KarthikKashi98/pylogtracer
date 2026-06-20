"""Unit tests for TimeResolver (deterministic via injected `now`)."""

from datetime import datetime

from pylogtracer.utils.time_resolver import TimeResolver


# Fixed anchor so relative phrases resolve deterministically.
NOW = datetime(2024, 3, 1, 15, 0, 0)


def _r(question):
    return TimeResolver(now=NOW).resolve(question)


def test_clock_time_anchors_to_today():
    res = _r("what errors happened at 10am?")
    assert res["resolved"] is True
    assert res["from_dt"] == "2024-03-01 10:00:00"
    assert res["to_dt"] == "2024-03-01 10:59:59"


def test_yesterday_full_day():
    res = _r("show errors from yesterday")
    assert res["date"] == "2024-02-29"
    assert res["from_dt"] is None and res["to_dt"] is None


def test_hours_ago_window():
    res = _r("what happened 2 hours ago?")
    assert res["from_dt"] == "2024-03-01 13:00:00"
    assert res["to_dt"] == "2024-03-01 15:00:00"


def test_absolute_datetime_passthrough():
    res = _r("errors at 2024-03-01 09:30:00 please")
    assert res["date"] == "2024-03-01"
    assert res["from_dt"] == "2024-03-01 09:30:00"


def test_no_time_reference_is_not_resolved():
    res = _r("is the system healthy?")
    assert res["resolved"] is False
    assert res["enriched_question"] == "is the system healthy?"
