"""Unit tests for the trust workstream: PII/secret redaction + agent grounding."""

from langchain_core.messages import HumanMessage, AIMessage

from pylogtracer import LogTracer
from pylogtracer.utils.redaction import redact
from pylogtracer.agents.qa_agent import QAAgent
from pylogtracer.preprocessing.error_type_classifier import (
    ErrorTypeClassifier,
    BatchClassification,
    EntryClassification,
)


# ── F1: redaction primitive ────────────────────────────────────────
def test_redact_masks_pii_and_secrets():
    text = ("login from 10.0.0.5 by bob@example.com using "
            "Bearer abc.def.ghi key sk-ABCDEFGHIJ1234567890")
    out = redact(text)
    assert "10.0.0.5" not in out
    assert "bob@example.com" not in out
    assert "abc.def.ghi" not in out
    assert "sk-ABCDEFGHIJ1234567890" not in out
    assert "<redacted-ip>" in out and "<redacted-email>" in out


def test_redact_leaves_plain_text_intact():
    text = "Worker process 3 crashed unexpectedly"
    assert redact(text) == text


# ── F1: auto policy on LogTracer ───────────────────────────────────
def test_redact_auto_off_for_ollama(sample_log_path):
    t = LogTracer(sample_log_path, llm_config={
        "provider": "ollama", "model": "qwen2.5:3b", "base_url": "http://localhost:11434"})
    assert t.redact is False
    assert t._redactor is None


def test_redact_auto_on_for_cloud(sample_log_path):
    t = LogTracer(sample_log_path, llm_config={
        "provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-test"})
    assert t.redact is True
    assert t._classifier.redactor is not None


def test_redact_force_override(sample_log_path):
    t = LogTracer(sample_log_path, llm_config={
        "provider": "ollama", "model": "x", "base_url": "http://localhost:11434"}, redact=True)
    assert t.redact is True


# ── F1: redaction actually applied at the classifier boundary ──────
class _FakeFactory:
    def get_model(self):
        return "qwen2.5:7b"

    def get_llm(self):  # pragma: no cover
        raise AssertionError("real LLM must not be built")


class _RecordingChain:
    def __init__(self):
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload["entries_text"])
        return BatchClassification(
            classifications={"1": EntryClassification(error_type="X", keywords=[])}
        )


def test_classifier_redacts_payload_before_llm():
    rec = _RecordingChain()
    c = ErrorTypeClassifier(factory=_FakeFactory(), redactor=redact)
    c._structured_llm = rec
    c.classify([{
        "primary_error": "login from 10.1.2.3 stalled for bob@corp.com",
        "error_type": "UnknownError", "is_duplicate": False, "type_source": "pending",
    }])
    sent = rec.calls[0]
    assert "10.1.2.3" not in sent and "bob@corp.com" not in sent


# ── F2: agent answer grounding ─────────────────────────────────────
def test_grounding_flags_fabricated_line():
    agent = QAAgent(tracer=None, factory=None)
    messages = [HumanMessage(content='TOOL_RESULT [search]:\n'
                             '{"entries": ["2024-01-01 10:00:00 ERROR real line here"]}')]
    answer = ("```\n"
              "2024-01-01 10:00:00 ERROR real line here\n"
              "2099-12-31 23:59:59 ERROR totally invented line\n"
              "```\n---\nsummary")
    grounded = agent._ground_answer(answer, messages)
    assert "could not be verified" in grounded


def test_grounding_passes_when_all_lines_real():
    agent = QAAgent(tracer=None, factory=None)
    messages = [HumanMessage(content='TOOL_RESULT [search]:\n'
                             '2024-01-01 10:00:00 ERROR real line here')]
    answer = "```\n2024-01-01 10:00:00 ERROR real line here\n```\n---\nok"
    grounded = agent._ground_answer(answer, messages)
    assert "could not be verified" not in grounded


def test_grounding_noop_without_tool_results():
    agent = QAAgent(tracer=None, factory=None)
    answer = "The system is healthy."
    assert agent._ground_answer(answer, []) == answer


def test_grounding_does_not_flag_prose_in_a_fence():
    # A correct prose/numeric answer wrapped in a fence (no timestamp lines)
    # must NOT be flagged — only quoted log lines are grounded.
    agent = QAAgent(tracer=None, factory=None)
    messages = [HumanMessage(content='TOOL_RESULT [keyword_duration]:\n'
                             '{"keyword": "REQ-4471", "duration_seconds": 4}')]
    answer = "```\nREQ-4471 lasted for 4 second(s).\n```"
    assert "could not be verified" not in agent._ground_answer(answer, messages)


def test_grounding_ignores_prose_that_mentions_a_timestamp():
    # The real qwen2.5:7b case: a CORRECT prose answer that mentions a timestamp
    # mid-sentence must not be flagged (only lines *starting* with a timestamp).
    agent = QAAgent(tracer=None, factory=None)
    messages = [HumanMessage(content='TOOL_RESULT [keyword_duration]:\n'
                             '{"first_occurrence": "2026-06-19 09:15:22", "duration_seconds": 4}')]
    answer = ("```\nThe incident REQ-4471 lasted for 4 second(s) from "
              "2026-06-19 09:15:22 to 2026-06-19 09:15:26.\n```")
    assert "could not be verified" not in agent._ground_answer(answer, messages)


# ── agent can dispatch the generic keyword_duration tool ───────────
def test_agent_dispatches_keyword_duration(sample_log_path):
    t = LogTracer(sample_log_path)
    agent = QAAgent(tracer=t, factory=None)
    out = agent._execute_tool("keyword_duration", {"keyword": "INC1000001"})
    assert out["found"] is True
    assert out["duration_seconds"] == 25


# ── Feature 1: evidence-based answers ──────────────────────────────
def test_evidence_attached_from_search_result():
    agent = QAAgent(tracer=None, factory=None, evidence=True)
    state = {"tool_evidence": [("search", {
        "keyword": "INC1", "total_found": 2,
        "entries": ["2024-01-01 10:00:00 ERROR boom INC1",
                    "2024-01-01 10:00:01 INFO ok INC1"],
    })]}
    out = agent._attach_evidence("Two entries found.", state)
    assert "Evidence (from logs):" in out
    assert "2024-01-01 10:00:00 ERROR boom INC1" in out


def test_evidence_keyword_duration_summary():
    agent = QAAgent(tracer=None, factory=None, evidence=True)
    state = {"tool_evidence": [("keyword_duration", {
        "keyword": "REQ-1", "occurrences": 3,
        "first_occurrence": "2026-06-19 09:15:22",
        "last_occurrence": "2026-06-19 09:15:26",
        "duration_human": "4 second(s)",
    })]}
    out = agent._attach_evidence("It lasted a little while.", state)
    assert "REQ-1: 3 occurrence(s)" in out
    assert "4 second(s)" in out


def test_evidence_can_be_disabled():
    agent = QAAgent(tracer=None, factory=None, evidence=False)
    state = {"tool_evidence": [("search", {"entries": ["some line"]})]}
    assert agent._attach_evidence("answer", state) == "answer"


# ── routing: gather evidence before answering ──────────────────────
def test_router_runs_tool_even_if_final_answer_present_without_evidence():
    # Model emitted BOTH a tool call and a FINAL_ANSWER and no tool has run yet
    # → must run the tool first (otherwise it answers with zero evidence).
    agent = QAAgent(tracer=None, factory=None)
    msg = AIMessage(content='TOOL: search\nARGS: {"keyword":"x"}\nFINAL_ANSWER:\n```\nfoo\n```')
    state = {"messages": [msg], "tool_evidence": [], "steps_taken": 0, "current_answer": None}
    assert agent._route_after_think(state) == "tool"


def test_router_finalizes_once_evidence_exists():
    agent = QAAgent(tracer=None, factory=None)
    msg = AIMessage(content='TOOL: search\nFINAL_ANSWER:\n```\nfoo\n```')
    state = {"messages": [msg], "tool_evidence": [("search", {})],
             "steps_taken": 1, "current_answer": None}
    assert agent._route_after_think(state) == "finalize"


def test_evidence_skips_lines_already_in_answer():
    agent = QAAgent(tracer=None, factory=None, evidence=True)
    line = "2024-01-01 10:00:00 ERROR boom"
    state = {"tool_evidence": [("search", {"entries": [line]})]}
    # The answer already quotes the line, so no duplicate Evidence block.
    out = agent._attach_evidence(f"```\n{line}\n```", state)
    assert "Evidence (from logs):" not in out
