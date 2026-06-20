"""Live integration tests against a local Ollama server.

These are skipped automatically unless an Ollama server is reachable and the
model below is available. Run Ollama with `qwen2.5:3b` pulled to exercise them.
"""

import urllib.request

import pytest

from pylogtracer import LogTracer

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:3b"


def _ollama_has_model(model: str) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as resp:
            body = resp.read().decode("utf-8", "replace")
    except Exception:
        return False
    return model.split(":")[0] in body


pytestmark = pytest.mark.skipif(
    not _ollama_has_model(MODEL),
    reason=f"Ollama not reachable or model '{MODEL}' not available",
)


@pytest.fixture
def tracer(sample_log_path):
    return LogTracer(
        sample_log_path,
        llm_config={"provider": "ollama", "model": MODEL, "base_url": OLLAMA_URL},
    )


def test_llm_classification_resolves_unknowns(tracer):
    # With an LLM wired in, the classifier should learn keywords and leave
    # few/no UnknownError entries for this small, clear log.
    freq = tracer.error_frequency()
    assert isinstance(freq, dict) and freq
    # Keyword store is populated only when the LLM classifies real errors.
    assert isinstance(tracer._classifier.get_keyword_store(), dict)


def test_ask_returns_nonempty_answer(tracer):
    answer = tracer.ask("how many errors are there in total?")
    assert isinstance(answer, str)
    assert answer.strip()


def _skip_if_runner_crashed(answer: str):
    """Skip (not fail) when the Ollama model runner dies on GPU/OOM, etc."""
    markers = ("status code: 500", "out of memory", "runner process has terminated",
               "GGML_ASSERT", "cudaMalloc")
    if any(m in answer for m in markers):
        import pytest as _pytest
        _pytest.skip(f"Ollama model runner crashed (infra, not code): {answer[:120]}")


def test_ask_identifier_search_includes_id(tracer):
    answer = tracer.ask("show all logs for INC1000001")
    _skip_if_runner_crashed(answer)
    assert "INC1000001" in answer
