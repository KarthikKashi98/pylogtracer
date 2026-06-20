"""Unit tests for ErrorTypeClassifier.

Covers the offline (no-LLM) paths, the regression test for the batch-remap
bug, and the 0.2.0 additions: within-run dedup, persistent keyword store,
and the token-budget guardrail.
"""

import os

import pytest

from pylogtracer.preprocessing.error_type_classifier import (
    ErrorTypeClassifier,
    BatchClassification,
    EntryClassification,
)


class _FakeChain:
    """Stand-in for the structured-output LCEL chain."""

    def __init__(self, result):
        self._result = result

    def invoke(self, _payload):
        return self._result


class _FakeFactory:
    """Minimal factory so classify() reaches the (mocked) LLM path."""

    def get_model(self):
        return "qwen2.5:7b"

    def get_llm(self):  # pragma: no cover - must never be called in tests
        raise AssertionError("real LLM should not be built in unit tests")


class _RecordingChain:
    """Records the entries_text sent and returns one classification per line."""

    def __init__(self, keywords=None):
        self.calls = []
        self._keywords = keywords or []

    def invoke(self, payload):
        text = payload["entries_text"]
        self.calls.append(text)
        n = len([ln for ln in text.splitlines() if ln.strip()])
        cls = {
            str(i + 1): EntryClassification(error_type=f"Type{i + 1}", keywords=self._keywords)
            for i in range(n)
        }
        return BatchClassification(classifications=cls)


def _entry(primary):
    return {
        "primary_error": primary,
        "error_type": "UnknownError",
        "is_duplicate": False,
        "type_source": "pending",
    }


def test_regex_classifies_named_exception():
    c = ErrorTypeClassifier(factory=None)
    out = c.classify([_entry("ValueError: bad input")])
    assert out[0]["error_type"] == "ValueError"
    assert out[0]["type_source"] == "regex"


def test_pattern_matcher_classifies_without_llm():
    c = ErrorTypeClassifier(factory=None)
    out = c.classify([_entry("connection refused by host")])
    assert out[0]["error_type"] == "ConnectionError"


def test_offline_unknown_stays_unknown_and_never_calls_network():
    # factory=None must not raise and must not attempt any LLM call.
    c = ErrorTypeClassifier(factory=None)
    out = c.classify([_entry("some totally opaque message xyzzy")])
    assert out[0]["error_type"] == "UnknownError"


def test_classify_batch_numeric_keys_pass_through():
    c = ErrorTypeClassifier(factory=None)
    batch = [_entry("disk full"), _entry("token expired")]
    c._structured_llm = _FakeChain(
        BatchClassification(
            classifications={
                "1": EntryClassification(error_type="DiskError", keywords=["disk full"]),
                "2": EntryClassification(error_type="AuthError", keywords=["token expired"]),
            }
        )
    )
    result = c._classify_batch(batch)
    assert result["1"].error_type == "DiskError"
    assert result["2"].error_type == "AuthError"


def test_classify_batch_nonnumeric_keys_map_positionally():
    """Regression: INC-style keys must map to distinct entries by position.

    Before the fix, the remap loop broke after the first item, assigning the
    first classification to EVERY entry in the batch.
    """
    c = ErrorTypeClassifier(factory=None)
    batch = [_entry("disk full"), _entry("token expired")]
    c._structured_llm = _FakeChain(
        BatchClassification(
            classifications={
                "INC1": EntryClassification(error_type="DiskError", keywords=["disk full"]),
                "INC2": EntryClassification(error_type="AuthError", keywords=["token expired"]),
            }
        )
    )
    result = c._classify_batch(batch)
    assert result["1"].error_type == "DiskError"
    assert result["2"].error_type == "AuthError"  # would be "DiskError" before the fix


def test_keyword_validation_rejects_ports_and_stopwords():
    c = ErrorTypeClassifier(factory=None)
    valid = c._validate_keywords(["db:5432", "the", "404", "email delivery"])
    assert "email delivery" in valid
    assert "db:5432" not in valid
    assert "the" not in valid
    assert "404" not in valid


# ── Workstream A: within-run dedup ─────────────────────────────────
def test_dedup_sends_only_unique_lines():
    c = ErrorTypeClassifier(factory=_FakeFactory())
    rec = _RecordingChain()
    c._structured_llm = rec
    entries = [_entry("opaque alpha glitch") for _ in range(3)] + \
              [_entry("opaque beta glitch") for _ in range(2)]
    out = c.classify(entries)

    assert len(rec.calls) == 1
    sent_lines = [ln for ln in rec.calls[0].splitlines() if ln.strip()]
    assert len(sent_lines) == 2                       # 5 entries -> 2 unique sent
    assert all(e["error_type"].startswith("Type") for e in out)
    assert sum(1 for e in out if e["is_duplicate"]) == 3  # the 3 duplicates flagged


def test_dedup_flat_as_log_grows():
    """Core-thesis regression: unique payload stays flat regardless of count."""
    for n in (2, 50):
        c = ErrorTypeClassifier(factory=_FakeFactory())
        rec = _RecordingChain()
        c._structured_llm = rec
        entries = [_entry("opaque alpha glitch") for _ in range(n)] + \
                  [_entry("opaque beta glitch") for _ in range(n)]
        c.classify(entries)
        sent = [ln for ln in "\n".join(rec.calls).splitlines() if ln.strip()]
        assert len(sent) == 2, f"expected 2 unique lines for n={n}, got {len(sent)}"


def test_dedup_disabled_sends_everything():
    c = ErrorTypeClassifier(factory=_FakeFactory(), dedup=False)
    rec = _RecordingChain()
    c._structured_llm = rec
    entries = [_entry("opaque alpha glitch") for _ in range(4)]
    c.classify(entries)
    sent = [ln for ln in "\n".join(rec.calls).splitlines() if ln.strip()]
    assert len(sent) == 4


def test_dedup_strip_ids_collapses_id_variants():
    c = ErrorTypeClassifier(factory=_FakeFactory(), dedup_strip_ids=True)
    rec = _RecordingChain()
    c._structured_llm = rec
    entries = [_entry("payment gateway exploded - INC1000001"),
               _entry("payment gateway exploded - INC1000002")]
    c.classify(entries)
    sent = [ln for ln in "\n".join(rec.calls).splitlines() if ln.strip()]
    assert len(sent) == 1   # collapsed despite different incident ids


# ── Workstream A: persistent keyword store ─────────────────────────
def test_keyword_store_persists_across_runs(tmp_path):
    cache = str(tmp_path / "kw.json")

    c1 = ErrorTypeClassifier(factory=_FakeFactory(), cache_path=cache)
    c1._structured_llm = _RecordingChain(keywords=["alpha glitch"])
    c1.classify([_entry("system alpha glitch detected")])
    c1.save()
    assert os.path.exists(cache)

    # A fresh classifier loads the cache and types a matching line for FREE.
    c2 = ErrorTypeClassifier(factory=_FakeFactory(), cache_path=cache)
    c2._structured_llm = _FakeChain(None)  # would crash if invoked
    out = c2.classify([_entry("another alpha glitch occurrence")])
    assert out[0]["type_source"] == "keyword"
    assert out[0]["error_type"] in c2.get_keyword_store()


def test_corrupt_cache_does_not_crash(tmp_path):
    cache = tmp_path / "kw.json"
    cache.write_text("{ this is not valid json", encoding="utf-8")
    c = ErrorTypeClassifier(factory=None, cache_path=str(cache))  # must not raise
    assert c.get_keyword_store() == {}


# ── Workstream B: token-budget guardrail ───────────────────────────
def test_max_context_tokens_caps_batch_size():
    c = ErrorTypeClassifier(factory=_FakeFactory(), max_context_tokens=600)
    # usable = max(1, 600-400)=200 -> floor(200/60)=3 -> clamp >= MIN_BATCH_SIZE(3)
    assert c._compute_batch_size() == 3


def test_max_entry_chars_truncates_payload_not_entry():
    c = ErrorTypeClassifier(factory=_FakeFactory(), max_entry_chars=10)
    rec = _RecordingChain()
    c._structured_llm = rec
    long = "x" * 100 + " unique-tail"
    e = _entry(long)
    c.classify([e])

    sent_line = rec.calls[0]
    assert "…" in sent_line                       # payload was truncated
    assert e["primary_error"] == long             # stored entry untouched


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
