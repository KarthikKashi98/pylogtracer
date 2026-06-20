"""
qa_agent.py
============
Dynamic ReAct agent for LogTracer.ask() — with multi-question support.

Architecture — LangGraph ReAct loop with question splitter:

    [START]
       ↓
    [split_questions]  ← LLM splits prompt into ordered sub-questions
       ↓
    [time_resolve]     ← resolve relative timestamps for current sub-question
       ↓
    [think]  ← LLM decides: which tool? or am I done?
       ↓  ↑
    [tool]   ← executes tool, result appended to message history
       ↓  ↑______ loop back if LLM wants more tools
    [finalize]         ← extract sub-answer from message history
       ↓
    [context_accumulator] ← store sub-answer; inject into next iteration
       ↓  ↑______________ loop back if more sub-questions remain
    [assemble_answers] ← stitch sub-answers sequentially, pure Python, no LLM
       ↓
    [END]

Output format for multiple sub-questions:
    ### Q: <sub-question 1>
    <answer 1>

    ### Q: <sub-question 2>
    <answer 2>

Max steps per sub-question: 8 (prevents infinite loops)

Usage:
    from agents.qa_agent import QAAgent
    agent  = QAAgent(tracer=tracer, factory=factory)
    answer = agent.run("show INC1033234 and how long did it last?")
"""

import json
import re
import logging
from typing import TypedDict, Optional, List, Any

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, START, END
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

MAX_STEPS = 8  # max tool calls per sub-question


# ── Agent state ───────────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    sub_questions: List[dict]
    current_index: int
    prior_answers: List[dict]
    messages: List[Any]
    steps_taken: int
    current_answer: Optional[str]
    final_answer: Optional[str]
    tool_evidence: List[Any]  # (tool_name, result) collected for this sub-question


# ── System prompts ────────────────────────────────────────────────

SPLITTER_PROMPT = """You are a question-splitting assistant.

The user may ask one or more questions in a single prompt. Your job is to:
1. Split the prompt into individual sub-questions.
2. Rewrite each sub-question as a clear, standalone sentence.
3. Order them logically.

==================================================
DEPENDENCY RULE (STRICT)

depends_on MUST be null unless the answer to another sub-question is literally
required as INPUT to answer this one.

Ask yourself:
"Can I answer this RIGHT NOW using only the entity/ID in the question itself?"
- YES → depends_on: null
- NO  → depends_on: <id of required question>

Sharing the same keyword or ID does NOT create a dependency.
Only a true data dependency (needing the other answer as input) does.

==================================================
CONTEXT PROPAGATION RULE

Include the entity (incident ID, keyword, etc.) explicitly in EVERY sub-question.
Never produce vague questions like "what are the logs?" or "why did it happen?".
Always produce self-contained questions like "What are the logs for INC1000004?".

==================================================
REWRITING RULE FOR "LIST / SHOW / GET LOGS" QUESTIONS (CRITICAL)

When the user asks to list, show, or retrieve log entries for an identifier,
you MUST rewrite the question using this exact pattern:

  "Search for ALL log entries (including INFO, DEBUG, WARNING, ERROR) that contain [IDENTIFIER]."

Examples:
  User: "list all logs for INC1000004"
  Rewrite: "Search for ALL log entries (including INFO, DEBUG, WARNING, ERROR) that contain INC1000004."

  User: "show me logs related to INC1000004"
  Rewrite: "Search for ALL log entries (including INFO, DEBUG, WARNING, ERROR) that contain INC1000004."

==================================================
REWRITING RULE FOR "PREDICTION / SPECIFIC DATA" QUESTIONS

When the user asks for specific data within logs (prediction, status, result, value),
rewrite using:

  "Search for the prediction result of [IDENTIFIER] in the logs."
  "Find the [data] for [IDENTIFIER] in the logs."

==================================================
OUTPUT FORMAT (STRICT)

Respond ONLY with a JSON array. No preamble, no explanation, no markdown.

Each element:
  "id"         : integer starting at 0
  "question"   : rewritten standalone question (apply rewriting rules above)
  "depends_on" : null OR id of required prior question

Use double quotes, null not None, no trailing commas, valid JSON only.

==================================================
EXAMPLES

Input: "what is the prediction result of INC1000004 and list all logs for this incident"
Output:
[
  {"id": 0, "question": "Search for the prediction result of INC1000004 in the logs.", "depends_on": null},
  {"id": 1, "question": "Search for ALL log entries (including INFO, DEBUG, WARNING, ERROR) that contain INC1000004.", "depends_on": null}
]

Input: "show INC1033234 and how long did it last?"
Output:
[
  {"id": 0, "question": "Search for ALL log entries (including INFO, DEBUG, WARNING, ERROR) that contain INC1033234.", "depends_on": null},
  {"id": 1, "question": "How long did the incident INC1033234 last?", "depends_on": null}
]

Input: "What errors happened today and is the system healthy?"
Output:
[
  {"id": 0, "question": "What errors happened today?", "depends_on": null},
  {"id": 1, "question": "Is the system currently healthy?", "depends_on": null}
]

Input: "find the latest error then explain why it happened"
Output:
[
  {"id": 0, "question": "What is the most recent error in the logs?", "depends_on": null},
  {"id": 1, "question": "Why did the most recent error happen?", "depends_on": 0}
]

==================================================
"""


REACT_SYSTEM_PROMPT = """You are a log analysis agent. Use tools to answer questions about logs.

==================================================
USING PRIOR CONTEXT

If the user message starts with "CONTEXT FROM PREVIOUS ANSWERS:", you are answering
a follow-up question. ALWAYS use that context:
  - Reference entities (incident IDs, error types, timestamps) from prior answers
  - Build on prior results — do not start from scratch
  - When searching, reuse identifiers found in prior answers

==================================================
TOOLS:
  search(keyword)                            — finds ALL log entries: INFO, DEBUG, WARNING, ERROR, CRITICAL
  get_related_logs(identifier)               — finds all related logs in same cluster
  get_entry_details(identifier)              — full details + traceback for one entry
  error_frequency(date?, from_dt?, to_dt?)   — count errors by type
  errors_by_date(date)                       — all errors on a specific date
  errors_in_range(from_dt, to_dt)            — errors between two timestamps
  last_incident()                            — most recent error cluster
  summary(date?, from_dt?, to_dt?)           — high-level log overview
  root_cause(date?, from_dt?, to_dt?)        — LLM root cause analysis
  health_check()                             — is the system healthy?
  incident_duration(date?, from_dt?, to_dt?) — how long did the most recent error burst last?
  keyword_duration(keyword)                  — how long ANY keyword lasted: time from its
                                               first to its last occurrence (incident id,
                                               trace id, path, error phrase — anything)

==================================================
TOOL SELECTION — KEYWORD TRIGGER TABLE

  Question contains...                          | Use this tool
  ----------------------------------------------|---------------------------
  "search", "ALL entries", "all logs",          |
  "including INFO", "find entries",             | search(keyword)
  "prediction", "find mentions", "show entries" |
  ----------------------------------------------|---------------------------
  "error cluster", "errors only",               |
  "which errors", "error analysis"              | get_related_logs(identifier)
  ----------------------------------------------|---------------------------
  "full details", "traceback",                  |
  "stack trace", "details of"                   | get_entry_details(identifier)
  ----------------------------------------------|---------------------------
  "how long", "duration", "lasted" + a SPECIFIC |
  keyword/id/path (e.g. "how long did X last")  | keyword_duration(keyword)
  ----------------------------------------------|---------------------------
  "how long", "duration", "lasted" with NO      |
  keyword (e.g. "how long did it last")         | incident_duration()
  ----------------------------------------------|---------------------------
  "healthy", "health", "status"                 | health_check()
  ----------------------------------------------|---------------------------
  "summary", "overview", "how many total"       | summary()
  ----------------------------------------------|---------------------------
  "root cause", "why did", "cause of"           | root_cause()
  ----------------------------------------------|---------------------------
  "error count", "frequency", "how many errors" | error_frequency()

DEFAULT RULE:
  When the question mentions an identifier and asks to "list", "show", "find",
  or "get" logs → ALWAYS use search(identifier). It finds everything.
  Use get_related_logs() ONLY when the question is specifically about the error cluster.

==================================================
HOW TO CALL A TOOL:
TOOL: tool_name
ARGS: {"key": "value"}
REASON: one line why

HOW TO GIVE FINAL ANSWER:
FINAL_ANSWER:
```
<paste log lines here exactly as returned — do not change any word>
```
---
<2-3 sentence summary here, only if helpful>

==================================================
RULES:
- One tool at a time
- Never change, rephrase, or add to log lines
- Log lines go inside ``` ``` exactly as the tool returned them
- Summary goes AFTER --- only
- If no logs found, say "No logs found."

==================================================
EXAMPLE 1 — search by identifier:

User: Search for ALL log entries (including INFO, DEBUG, WARNING, ERROR) that contain INC1000004.

TOOL: search
ARGS: {"keyword": "INC1000004"}
REASON: search finds all log types for this identifier

[tool returns]
2024-01-15 10:23:41 ERROR [AuthService] Token validation failed — INC1000004
2024-01-15 10:23:45 WARN  [Gateway] Retrying request — INC1000004
2024-01-15 10:23:46 INFO  [Gateway] Retry successful — INC1000004

FINAL_ANSWER:
```
2024-01-15 10:23:41 ERROR [AuthService] Token validation failed — INC1000004
2024-01-15 10:23:45 WARN  [Gateway] Retrying request — INC1000004
2024-01-15 10:23:46 INFO  [Gateway] Retry successful — INC1000004
```
---
3 log entries found for INC1000004. AuthService reported a token failure; gateway retried and recovered.

==================================================
EXAMPLE 2 — nothing found:

User: Search for ALL log entries that contain REQ-xyz999.

TOOL: search
ARGS: {"keyword": "REQ-xyz999"}
REASON: search all log types for this identifier

[tool returns]
(empty)

FINAL_ANSWER:
No logs found for REQ-xyz999.
==================================================
"""


class QAAgent:
    """
    Dynamic ReAct agent with multi-question support.

    Sub-questions are answered independently and displayed sequentially —
    no merge LLM, no risk of log line hallucination during combination.

    Args:
        tracer:  LogTracer instance
        factory: LLMFactory instance
    """

    def __init__(self, tracer, factory, redactor=None, evidence=True):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph not installed. Run: pip install pylogtracer[agent]"
            )
        self.tracer = tracer
        self.factory = factory
        self.redactor = redactor  # optional callable(text)->text for the LLM boundary
        self.evidence = evidence  # append verifiable log-line evidence to answers
        self._graph = None

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def run(self, question: str) -> str:
        """Answer any free-form question (or multiple questions) about the logs."""
        graph = self._get_graph()
        state = graph.invoke(
            {
                "question": question,
                "sub_questions": [],
                "current_index": 0,
                "prior_answers": [],
                "messages": [],
                "steps_taken": 0,
                "current_answer": None,
                "final_answer": None,
                "tool_evidence": [],
            }
        )
        return state["final_answer"] or "I could not find an answer."

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — nodes
    # ─────────────────────────────────────────────────────────────

    def _node_split_questions(self, state: AgentState) -> AgentState:
        """Split the user prompt into sub-questions using the SPLITTER_PROMPT."""
        logger.debug("[QAAgent] Splitting question...")
        try:
            llm = self.factory.get_llm()
            response = llm.invoke(
                [
                    SystemMessage(content=SPLITTER_PROMPT),
                    HumanMessage(content=state["question"]),
                ]
            )
            raw = response.content if hasattr(response, "content") else str(response)
            logger.debug("[QAAgent] Splitter raw  : %r", raw[:400])
            cleaned = self._clean_json_output(raw)
            logger.debug("[QAAgent] Splitter clean: %r", cleaned[:400])
            sub_questions = json.loads(cleaned)

            if not isinstance(sub_questions, list) or not sub_questions:
                raise ValueError("Empty or invalid split result")

            logger.debug("[QAAgent] Split into %d sub-question(s)", len(sub_questions))
            for sq in sub_questions:
                dep = (
                    f" (depends on Q{sq['depends_on']})"
                    if sq.get("depends_on") is not None
                    else ""
                )
                logger.debug("    Q%s: %s%s", sq["id"], sq["question"], dep)

        except Exception as e:
            logger.warning("[QAAgent] Split failed (%s), treating as single question", e)
            sub_questions = [
                {"id": 0, "question": state["question"], "depends_on": None}
            ]

        return {**state, "sub_questions": sub_questions, "current_index": 0}

    def _node_time_resolve(self, state: AgentState) -> AgentState:
        """
        Resolve relative timestamps for the current sub-question.
        Builds fresh message history, injecting prior answers as context.
        """
        from pylogtracer.utils.time_resolver import TimeResolver
        logger.debug("[QAAgent] Resolving time for current sub-question...")
        sq = state["sub_questions"][state["current_index"]]
        resolved = TimeResolver().resolve(sq["question"])
        enriched = resolved["enriched_question"]

        if resolved["resolved"]:
            logger.debug(
                "[QAAgent] Q%s time resolved: from=%s | to=%s | date=%s",
                sq["id"], resolved["from_dt"], resolved["to_dt"], resolved["date"],
            )

        context_block = ""
        if state["prior_answers"]:
            lines = ["CONTEXT FROM PREVIOUS ANSWERS:"]
            for pa in state["prior_answers"]:
                lines.append(f"Q: {pa['question']}")
                lines.append(f"A: {pa['answer']}")
                lines.append("")
            context_block = "\n".join(lines) + "\n"

        user_content = f"{context_block}Now answer this question:\n{enriched}"

        return {
            **state,
            "messages": [
                SystemMessage(content=REACT_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ],
            "steps_taken": 0,
            "current_answer": None,
            "tool_evidence": [],  # fresh evidence per sub-question
        }

    def _node_context_accumulator(self, state: AgentState) -> AgentState:
        """Store current sub-answer and advance the sub-question index."""
        logger.debug("[QAAgent] Accumulating context for next question...")
        sq = state["sub_questions"][state["current_index"]]
        answer = state.get("current_answer") or "No answer found."

        updated_prior = state["prior_answers"] + [
            {"question": sq["question"], "answer": answer}
        ]
        logger.debug("[QAAgent] Accumulated answer for Q%s (%d chars)", sq["id"], len(answer))

        return {
            **state,
            "prior_answers": updated_prior,
            "current_index": state["current_index"] + 1,
            "current_answer": None,
        }

    def _node_assemble_answers(self, state: AgentState) -> AgentState:
        """
        Assemble all sub-answers into the final response — pure Python, no LLM.

        Single sub-question  → return answer as-is, no header.
        Multiple sub-questions → each shown under its question as a header:

            ### Q: <question 1>
            <answer 1>

            ### Q: <question 2>
            <answer 2>

        Raw log lines are never touched — zero paraphrasing risk.
        """
        logger.debug("[QAAgent] Assembling final answer from sub-answers...")
        prior = state["prior_answers"]

        if len(prior) == 1:
            return {**state, "final_answer": prior[0]["answer"]}

        sections = []
        for pa in prior:
            sections.append(f"### Q: {pa['question']}\n{pa['answer']}")

        return {**state, "final_answer": "\n\n".join(sections)}

    def _node_think(self, state: AgentState) -> AgentState:
        """LLM decides: call a tool (TOOL:) or give the final answer (FINAL_ANSWER:)."""
        logger.debug("[QAAgent] Thinking...")
        if state["steps_taken"] >= MAX_STEPS:
            sq = state["sub_questions"][state["current_index"]]
            logger.debug("[QAAgent] Q%s: max steps reached — forcing answer", sq["id"])
            summary = self._summarize_results(state["messages"])
            return {**state, "current_answer": f"Based on what I found:\n\n{summary}"}

        try:
            llm = self.factory.get_llm()
            response = llm.invoke(state["messages"])
            content = response.content if hasattr(response, "content") else str(response)
            sq = state["sub_questions"][state["current_index"]]
            preview = content[:100].replace("\n", " ").strip()
            logger.debug("[QAAgent] Q%s step %d: %s...", sq["id"], state["steps_taken"] + 1, preview)
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content=content)],
            }
        except Exception as e:
            logger.error("[QAAgent] LLM error: %s", e)
            return {**state, "current_answer": f"Error during analysis: {e}"}

    def _node_tool(self, state: AgentState) -> AgentState:
        """Parse TOOL/ARGS from last AI message, execute tool, append result."""
        last_content = self._last_ai_content(state["messages"])
        tool_name, tool_args = self._parse_tool_call(last_content)
        sq = state["sub_questions"][state["current_index"]]
        logger.debug("[QAAgent] Q%s tool: %s(%s)", sq["id"], tool_name, tool_args)

        try:
            result = self._execute_tool(tool_name, tool_args)
        except Exception as e:
            result = {"error": f"Tool failed: {e}"}
            logger.error("[QAAgent] Tool error: %s", e)

        result_text = (
            f"TOOL_RESULT [{tool_name}]:\n"
            f"{json.dumps(result, indent=2, default=str)}"
        )

        # Scrub PII/secrets right before tool output enters the LLM transcript.
        if self.redactor:
            result_text = self.redactor(result_text)

        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content=result_text)],
            "steps_taken": state["steps_taken"] + 1,
            "tool_evidence": state.get("tool_evidence", []) + [(tool_name, result)],
        }

    def _node_finalize(self, state: AgentState) -> AgentState:
        """Extract FINAL_ANSWER from message history into current_answer."""
        logger.debug("[QAAgent] Finalizing answer from message history...")
        if state.get("current_answer"):
            return state

        answer = None
        for msg in reversed(state["messages"]):
            content = msg.content if hasattr(msg, "content") else ""
            if "FINAL_ANSWER:" in content:
                extracted = self._extract_final_answer(content)
                if extracted is not None:
                    answer = self._ground_answer(extracted, state["messages"])
                    break

        if answer is None:  # fallback: last AI message
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

        if answer is None:
            answer = "No answer generated."

        answer = self._attach_evidence(answer, state)
        return {**state, "current_answer": answer}

    def _attach_evidence(self, answer: str, state: AgentState) -> str:
        """Append the actual tool-sourced log lines as verifiable evidence."""
        if not self.evidence:
            return answer
        lines = self._collect_evidence_lines(state.get("tool_evidence") or [])
        # Don't repeat lines the answer already quotes.
        lines = [ln for ln in lines if ln and ln not in answer]
        if not lines:
            return answer
        block = "\n".join("- " + ln for ln in lines)
        return f"{answer}\n\nEvidence (from logs):\n{block}"

    def _collect_evidence_lines(self, tool_evidence: list) -> list:
        """Pull a compact, factual evidence list from the raw tool results."""
        out = []
        for _name, result in tool_evidence:
            if isinstance(result, dict):
                if isinstance(result.get("entries"), list):
                    out += [e.splitlines()[0] if isinstance(e, str) else str(e)
                            for e in result["entries"][:6]]
                elif isinstance(result.get("all_entries"), list):
                    out += [e.splitlines()[0] if isinstance(e, str) else str(e)
                            for e in result["all_entries"][:6]]
                elif "first_occurrence" in result:        # keyword_duration
                    out.append(f"{result.get('keyword')}: {result.get('occurrences')} occurrence(s), "
                               f"{result.get('first_occurrence')} -> {result.get('last_occurrence')} "
                               f"({result.get('duration_human')})")
                elif "duration_human" in result:          # incident_duration
                    out.append(f"incident lasted {result.get('duration_human')} "
                               f"({result.get('start')} -> {result.get('end')})")
                elif "status" in result and "summary" in result:  # health_check
                    out.append(result["summary"])
                elif "root_cause" in result:              # root cause
                    out.append("root cause: " + str(result.get("root_cause", "")))
                else:                                      # frequency / generic mapping
                    items = list(result.items())[:8]
                    out.append(", ".join(f"{k}={v}" for k, v in items))
            elif isinstance(result, list):                # list of error dicts
                for e in result[:6]:
                    if isinstance(e, dict):
                        out.append(f"{e.get('timestamp')} {e.get('error_type')} "
                                   f"{str(e.get('primary_error', ''))[:70]}")
                    else:
                        out.append(str(e)[:120])
        # De-dup preserving order, cap total.
        seen, deduped = set(), []
        for ln in out:
            if ln and ln not in seen:
                seen.add(ln)
                deduped.append(ln)
        return deduped[:10]

    def _ground_answer(self, answer: str, messages: list) -> str:
        """
        Verify quoted LOG LINES against the tool output collected this turn.

        Only lines that look like real log lines (i.e. carry a timestamp) are
        checked — prose sentences and numeric summaries are left alone, so a
        correct answer is never falsely flagged. A timestamped line that does
        NOT appear verbatim in any TOOL_RESULT is likely hallucinated, so we
        append a plain-ASCII note (never alter or drop content).
        """
        from pylogtracer.preprocessing import log_format

        def _starts_with_timestamp(s: str) -> bool:
            # A quoted LOG line begins with a timestamp; prose only mentions one
            # mid-sentence. Anchoring at the start avoids flagging correct prose.
            return any(re.match(r"\s*" + pat, s) for pat, _fmt in log_format.TS_PATTERNS)

        # Build the corpus of everything the tools returned this turn.
        corpus = "\n".join(
            (m.content if isinstance(m.content, str) else str(m.content))
            for m in messages
            if hasattr(m, "content") and "TOOL_RESULT" in str(m.content)
        )
        if not corpus:
            return answer  # no tools were called — nothing to ground against

        fence = re.search(r"```(.*?)```", answer, re.DOTALL)
        if not fence:
            return answer

        unverified = []
        for line in fence.group(1).splitlines():
            stripped = line.strip()
            if len(stripped) < 12:
                continue  # skip blanks / trivial lines
            if not _starts_with_timestamp(stripped):
                continue  # not a quoted log line (prose/summary) — don't ground it
            if stripped not in corpus:
                unverified.append(stripped)

        if unverified:
            logger.warning("[QAAgent] %d answer line(s) not found in tool output", len(unverified))
            answer += (
                "\n\nNote: {} log line(s) above could not be verified against the "
                "search results and may be inaccurate.".format(len(unverified))
            )
        return answer

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — routing
    # ─────────────────────────────────────────────────────────────

    def _route_after_think(self, state: AgentState) -> str:
        logger.debug("[QAAgent] Routing after think...")
        if state.get("current_answer"):
            return "finalize"
        content = self._last_ai_content(state["messages"])
        has_tool = "TOOL:" in content
        has_final = "FINAL_ANSWER:" in content
        # Evidence-first: if the model proposes a tool and we have NOT run one
        # yet, execute it even when the model also slapped on a FINAL_ANSWER in
        # the same message. Otherwise it would "answer" with zero evidence —
        # exactly the path that let hallucinated log lines through.
        if has_tool and not state.get("tool_evidence"):
            return "tool"
        if has_final:
            return "finalize"
        if has_tool:
            return "tool"
        # Malformed output — retry if budget allows, else finalize
        if state["steps_taken"] < MAX_STEPS:
            logger.debug("[QAAgent] Malformed output — retrying think")
            return "tool"
        return "finalize"

    def _route_after_accumulate(self, state: AgentState) -> str:
        if state["current_index"] < len(state["sub_questions"]):
            return "next_question"
        return "assemble"

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — tool execution
    # ─────────────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, args: dict) -> Any:
        t = self.tracer

        if tool_name == "error_frequency":
            return t.error_frequency(**self._safe_args(args, ["date", "from_dt", "to_dt"]))

        elif tool_name == "errors_by_date":
            date = args.get("date")
            if not date:
                return {"error": "date required"}
            return self._fmt_errors(t.errors_by_date(date))

        elif tool_name == "errors_in_range":
            from_dt = args.get("from_dt")
            to_dt = args.get("to_dt")
            if not from_dt or not to_dt:
                return {"error": "from_dt and to_dt required"}
            return self._fmt_errors(t.errors_in_range(from_dt, to_dt))

        elif tool_name == "last_incident":
            return self._fmt_errors(t.last_incident())

        elif tool_name == "summary":
            return t.summary(**self._safe_args(args, ["date", "from_dt", "to_dt"]))

        elif tool_name == "root_cause":
            return t.root_cause_analysis(**self._safe_args(args, ["date", "from_dt", "to_dt"]))

        elif tool_name == "health_check":
            return t.health_check()

        elif tool_name == "incident_duration":
            return t.incident_duration(**self._safe_args(args, ["date", "from_dt", "to_dt"]))

        elif tool_name == "keyword_duration":
            kw = args.get("keyword") or args.get("identifier", "")
            if not kw:
                return {"error": "keyword required"}
            return t.keyword_duration(kw)

        elif tool_name == "search":
            kw = args.get("keyword") or args.get("identifier", "")
            return t.search(kw)

        elif tool_name == "get_related_logs":
            idf = args.get("identifier") or args.get("keyword", "")
            return t.get_related_logs(idf)

        elif tool_name == "get_entry_details":
            idf = args.get("identifier") or args.get("keyword", "")
            return t.get_entry_details(idf)

        else:
            return {
                "error": (
                    f"Unknown tool: {tool_name}. Available: error_frequency, "
                    f"errors_by_date, errors_in_range, last_incident, summary, "
                    f"root_cause, health_check, incident_duration, keyword_duration, "
                    f"search, get_related_logs, get_entry_details"
                )
            }

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — helpers
    # ─────────────────────────────────────────────────────────────

    def _clean_json_output(self, raw: str) -> str:
        """Repair common small-model JSON mistakes before calling json.loads()."""
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw).strip()
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        raw = re.sub(r",\s*([\]}])", r"\1", raw)
        raw = re.sub(r"\}\s*\n\s*\{", "}, {", raw)
        raw = re.sub(r":\s*None\b", ": null", raw)
        raw = re.sub(r"'([^']*)'", r'"\1"', raw)
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("}"):
            raw = raw[:-1] + "]"
        return raw

    def _parse_tool_call(self, content: str):
        """Parse TOOL name and ARGS dict from LLM response."""
        tool_m = re.search(r"TOOL:\s*(\w+)", content)
        args_m = re.search(r"ARGS:\s*(\{.*?\})", content, re.DOTALL)

        if not tool_m:
            return None, {}

        tool_name = tool_m.group(1).strip()
        args = {}

        if args_m:
            try:
                raw = args_m.group(1).strip()
                args = json.loads(raw)
                args = {
                    k: v
                    for k, v in args.items()
                    if v is not None and v != "null" and v != ""
                }
            except json.JSONDecodeError:
                pairs = re.findall(r'"(\w+)":\s*"([^"]+)"', args_m.group(1))
                args = dict(pairs)

        return tool_name, args

    def _extract_final_answer(self, content: str) -> Optional[str]:
        """Extract text after FINAL_ANSWER:, stripping any leaked prompt text."""
        if not isinstance(content, str):
            content = str(content)

        m = re.search(r"FINAL_ANSWER:\s*(.+)", content, re.DOTALL)
        if not m:
            return content.strip()

        answer = m.group(1).strip()

        cutoffs = [
            "\nIMPORTANT RULES:",
            "\nTOOL:",
            "\nREACT LOOP",
            "\nTOOLS AVAILABLE:",
            "\nRULES",
            "\n- ONE tool",
            "\n- Call tools ONE at a time",
            "\n- After each tool result",
            "\nyour complete answer here",
        ]
        for cutoff in cutoffs:
            idx = answer.find(cutoff)
            if idx != -1:
                answer = answer[:idx].strip()

        placeholder_patterns = [
            "<write your answer here",
            "<specific answer",
            "Provide only the final answer",
            "Provide your answer",
        ]
        for pattern in placeholder_patterns:
            if (
                pattern.lower() in answer.lower()
                and answer.count("<") > 0
                and answer.count(">") > 0
            ):
                return None

        return answer

    def _last_ai_content(self, messages: list) -> str:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content if hasattr(msg, "content") else str(msg)
                return str(content) if not isinstance(content, str) else content
        return ""

    def _safe_args(self, args: dict, allowed: list) -> dict:
        return {
            k: v
            for k, v in args.items()
            if k in allowed and v and v != "null"
        }

    def _fmt_errors(self, errors: list) -> list:
        return [
            {
                "timestamp": (
                    e["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                    if e.get("timestamp") else None
                ),
                "error_type": e.get("error_type"),
                "primary_error": e.get("primary_error"),
                "traceback": e.get("traceback", ""),
            }
            for e in errors
        ]

    def _summarize_results(self, messages: list) -> str:
        blocks = [
            msg.content
            for msg in messages
            if hasattr(msg, "content") and "TOOL_RESULT" in msg.content
        ]
        return "\n\n".join(blocks) if blocks else "No results collected."

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — LangGraph graph
    # ─────────────────────────────────────────────────────────────

    def _get_graph(self):
        """Build and cache the LangGraph."""
        if self._graph is not None:
            return self._graph

        builder = StateGraph(AgentState)

        builder.add_node("split_questions", self._node_split_questions)
        builder.add_node("time_resolve", self._node_time_resolve)
        builder.add_node("think", self._node_think)
        builder.add_node("tool", self._node_tool)
        builder.add_node("finalize", self._node_finalize)
        builder.add_node("context_accumulator", self._node_context_accumulator)
        builder.add_node("assemble_answers", self._node_assemble_answers)

        builder.add_edge(START, "split_questions")
        builder.add_edge("split_questions", "time_resolve")
        builder.add_edge("time_resolve", "think")

        builder.add_conditional_edges(
            "think",
            self._route_after_think,
            {"tool": "tool", "finalize": "finalize"},
        )
        builder.add_edge("tool", "think")
        builder.add_edge("finalize", "context_accumulator")

        builder.add_conditional_edges(
            "context_accumulator",
            self._route_after_accumulate,
            {
                "next_question": "time_resolve",
                "assemble": "assemble_answers",
            },
        )
        builder.add_edge("assemble_answers", END)

        self._graph = builder.compile()
        return self._graph
