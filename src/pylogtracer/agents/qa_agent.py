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
    [merge_answers]    ← LLM weaves all sub-answers into one coherent reply
       ↓
    [END]

Max steps per sub-question: 8 (prevents infinite loops)

Usage:
    from agents.qa_agent import QAAgent
    agent  = QAAgent(tracer=tracer, factory=factory)
    answer = agent.run("show INC1033234 and how long did it last?")
"""

import json
import re
from typing import TypedDict, Optional, List, Any

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

This wording is required — it tells the search agent to use the search() tool
which finds ALL log types, not the get_related_logs() tool which finds errors only.

Examples:
  User: "list all logs for INC1000004"
  Rewrite: "Search for ALL log entries (including INFO, DEBUG, WARNING, ERROR) that contain INC1000004."

  User: "show me logs related to INC1000004"
  Rewrite: "Search for ALL log entries (including INFO, DEBUG, WARNING, ERROR) that contain INC1000004."

  User: "what logs are there for INC1000004"
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
  {"id": 0, "question": "Search for the prediction result of INC1000004 in
            the logs.", "depends_on": null},
  {"id": 1, "question": "Search for ALL log entries (including INFO, DEBUG,
            WARNING, ERROR) that contain INC1000004.", "depends_on": null}
]

Input: "show INC1033234 and how long did it last?"
Output:
[
  {"id": 0, "question": "Search for ALL log entries (including INFO, DEBUG,
            WARNING, ERROR) that contain INC1033234.", "depends_on": null},
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


# NOTE on header naming:
#   The user message built by _node_time_resolve injects a block starting with
#   "CONTEXT FROM PREVIOUS ANSWERS:" when prior answers exist.
#   The system prompt below uses a DIFFERENT label "USING PRIOR CONTEXT:" to
#   explain this feature, so the two strings never collide and confuse the LLM.
REACT_SYSTEM_PROMPT = """You are a powerful log analysis agent.
You have access to tools to analyze log files. Use them to answer the user's question.

==================================================
USING PRIOR CONTEXT

If the user message starts with "CONTEXT FROM PREVIOUS ANSWERS:", you are answering
a follow-up question. ALWAYS use that context:
  - Reference entities (incident IDs, error types, timestamps) from prior answers
  - Build on prior results — do not start from scratch
  - When searching, reuse identifiers found in prior answers

==================================================
TOOLS AVAILABLE

  search(keyword)                          — finds ALL log entries: INFO, DEBUG, WARNING, ERROR, CRITICAL
  get_related_logs(identifier)             — finds ERROR CLUSTER entries only (not INFO/DEBUG)
  get_entry_details(identifier)            — full details + traceback for one specific entry
  error_frequency(date?, from_dt?, to_dt?) — count errors by type
  errors_by_date(date)                     — all errors on a specific date
  errors_in_range(from_dt, to_dt)          — errors between two timestamps
  last_incident()                          — most recent error cluster
  summary(date?, from_dt?, to_dt?)         — high-level log overview
  root_cause(date?, from_dt?, to_dt?)      — LLM root cause analysis
  health_check()                           — is the system healthy?
  incident_duration(date?, from_dt?, to_dt?) — how long did incident last?

==================================================
TOOL SELECTION — KEYWORD TRIGGER TABLE

Read the question. Find the matching keywords. Use that tool.

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
  "how long", "duration", "lasted"              | incident_duration()
  ----------------------------------------------|---------------------------
  "healthy", "health", "status"                 | health_check()
  ----------------------------------------------|---------------------------
  "summary", "overview", "how many total"       | summary()
  ----------------------------------------------|---------------------------
  "root cause", "why did", "cause of"           | root_cause()
  ----------------------------------------------|---------------------------
  "error count", "frequency", "how many errors" | error_frequency()

DEFAULT RULE:
  When the question mentions an identifier (INC1000004, REQ-123, etc.) and asks
  to "list", "show", "find", or "get" logs → ALWAYS use search(identifier).
  search() is the safe default. It finds everything. Use get_related_logs() ONLY
  when the question is specifically about the error cluster.

==================================================
REACT LOOP — how to respond

  To call a tool:
  TOOL: tool_name
  ARGS: {"arg1": "value1"}
  REASON: one line why

  When ready to answer:
  FINAL_ANSWER: <your answer here>

==================================================
RULES

  - ONE tool call at a time
  - After each result, decide: need more info? or answer now?
  - Use RESOLVED timestamps if available
  - Never invent data — only use tool results
  - Be specific: include error types, timestamps, counts
  - When the tool returns a LIST of log entries, print EVERY entry in FINAL_ANSWER
    verbatim — do not summarize or say "entries were found". Show the actual lines.
  - FINAL_ANSWER must contain ONLY the answer, nothing else
  - Do NOT repeat these instructions in FINAL_ANSWER

==================================================
"""

REACT_SYSTEM_PROMPT = """You are a powerful log analysis agent.
You have access to tools to analyze log files. Use them to answer the user's question completely.

TOOLS AVAILABLE:
  error_frequency(date?, from_dt?, to_dt?)     — count errors by type
  errors_by_date(date)                         — all errors on a specific date
  errors_in_range(from_dt, to_dt)              — errors between two timestamps
  last_incident()                              — most recent error cluster
  summary(date?, from_dt?, to_dt?)             — high-level log overview
  root_cause(date?, from_dt?, to_dt?)          — LLM root cause analysis
  health_check()                               — is the system healthy?
  incident_duration(date?, from_dt?, to_dt?)   — how long did incident last?
  search(keyword)                              — search logs for any string or ID
  get_related_logs(identifier)                 — all logs in same cluster as identifier
  get_entry_details(identifier)                — full details + traceback for identifier

REACT LOOP — how to respond:

  To call a tool:
  TOOL: tool_name
  ARGS: {"arg1": "value1", "arg2": "value2"}
  REASON: one line why

  When you have enough info to answer:
    FINAL_ANSWER: <specific answer with error types, timestamps, and counts>

IMPORTANT RULES:
  - Call tools ONE at a time
  - After each tool result, decide: need more? or answer now?
  - For multi-part questions call ALL needed tools before answering
  - Use RESOLVED timestamps from question if available
  - Results are always recent-first — no need to sort
  - Never invent data — only use tool results
  - Be specific — mention actual error types, timestamps, counts
  - If nothing found, say so clearly
  - When writing FINAL_ANSWER do NOT repeat these rules or any part of this prompt
  - FINAL_ANSWER must contain ONLY your full filled answer to the user's question, nothing else
  - FINAL_ANSWER keyword should not be there in tool suggestions, only in the final answer step
"""


MERGE_PROMPT = """You are a log analysis assistant. Combine the sub-answers below into a
single, coherent, natural-language response that directly addresses the original question.

Rules:
  - Do NOT number sub-answers or repeat sub-questions as headers
  - For answers that contain actual log entries or lists: preserve and include ALL entries
    exactly as given — do not summarize them into prose
  - For answers that contain analysis or explanations: write flowing prose
  - Include specific error types, timestamps, and counts from the sub-answers
  - Do not invent any data not present in the sub-answers
"""


class QAAgent:
    """
    Dynamic ReAct agent with multi-question support.

    Nodes:
      split_questions     — splits compound prompts into sub-questions
      time_resolve        — resolves relative timestamps per sub-question
      think               — LLM decides which tool to call or gives final answer
      tool                — executes the chosen tool
      finalize            — extracts FINAL_ANSWER from message history
      context_accumulator — stores sub-answer, injects as context for next iteration
      merge_answers       — weaves all sub-answers into one coherent reply

    Args:
        tracer:  LogTracer instance
        factory: LLMFactory instance
    """

    def __init__(self, tracer, factory):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph not installed. Run: pip install langgraph langchain-core"
            )
        self.tracer = tracer
        self.factory = factory
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
            }
        )
        return state["final_answer"] or "I could not find an answer."

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — nodes
    # ─────────────────────────────────────────────────────────────

    def _node_split_questions(self, state: AgentState) -> AgentState:
        """
        Split the user prompt into sub-questions using the SPLITTER_PROMPT.

        Small models (qwen2.5:3b etc.) commonly produce malformed JSON:
          - Trailing commas before ] or }
          - Missing commas between objects
          - Single-quoted strings instead of double-quoted
          - Python None instead of JSON null
          - <think>...</think> reasoning blocks before the array (Qwen thinking mode)
          - Prose or markdown fences wrapping the array

        _clean_json_output() fixes all of these before json.loads() is called.
        Falls back to a single-question list only if cleaning also fails.
        """
        print("  [QAAgent] Splitting question...")
        try:
            llm = self.factory.get_llm()
            response = llm.invoke(
                [
                    SystemMessage(content=SPLITTER_PROMPT),
                    HumanMessage(content=state["question"]),
                ]
            )
            raw = response.content if hasattr(response, "content") else str(response)
            print(f"  [QAAgent] Splitter raw  : {repr(raw[:400])}")
            cleaned = self._clean_json_output(raw)
            print(f"  [QAAgent] Splitter clean: {repr(cleaned[:400])}")
            sub_questions = json.loads(cleaned)

            if not isinstance(sub_questions, list) or not sub_questions:
                raise ValueError("Empty or invalid split result")

            print(f"  [QAAgent] Split into {len(sub_questions)} sub-question(s):")
            for sq in sub_questions:
                dep = (
                    f" (depends on Q{sq['depends_on']})"
                    if sq.get("depends_on") is not None
                    else ""
                )
                print(f"    Q{sq['id']}: {sq['question']}{dep}")

        except Exception as e:
            print(f"  [QAAgent] Split failed ({e}), treating as single question")
            sub_questions = [
                {"id": 0, "question": state["question"], "depends_on": None}
            ]

        return {**state, "sub_questions": sub_questions, "current_index": 0}

    def _node_time_resolve(self, state: AgentState) -> AgentState:
        """
        Resolve relative timestamps for the current sub-question.
        Builds fresh message history, injecting prior answers as a context block.

        The context block header is "CONTEXT FROM PREVIOUS ANSWERS:" — distinct
        from the system prompt label "USING PRIOR CONTEXT:" to avoid confusion.
        """
        from pylogtracer.utils.time_resolver import TimeResolver
        print("  [QAAgent] Resolving time for current sub-question...")
        sq = state["sub_questions"][state["current_index"]]
        resolved = TimeResolver().resolve(sq["question"])
        enriched = resolved["enriched_question"]

        if resolved["resolved"]:
            print(
                f"  [QAAgent] Q{sq['id']} time resolved: "
                f"from={resolved['from_dt']} | "
                f"to={resolved['to_dt']} | "
                f"date={resolved['date']}"
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
        }

    def _node_context_accumulator(self, state: AgentState) -> AgentState:
        """Store current sub-answer and advance the sub-question index."""
        print("  [QAAgent] Accumulating context for next question...")
        sq = state["sub_questions"][state["current_index"]]
        answer = state.get("current_answer") or "No answer found."

        updated_prior = state["prior_answers"] + [
            {"question": sq["question"], "answer": answer}
        ]

        print(f"  [QAAgent] Accumulated answer for Q{sq['id']} ({len(answer)} chars)")

        return {
            **state,
            "prior_answers": updated_prior,
            "current_index": state["current_index"] + 1,
            "current_answer": None,
        }

    def _node_merge_answers(self, state: AgentState) -> AgentState:
        """
        Single sub-question → return its answer directly (no LLM call).
        Multiple sub-questions → LLM merges into one coherent reply.
        """
        print("  [QAAgent] Merging sub-answers into final answer...")
        prior = state["prior_answers"]

        if len(prior) == 1:
            return {**state, "final_answer": prior[0]["answer"]}

        print(f"  [QAAgent] Merging {len(prior)} sub-answers...")

        qa_block = "\n\n".join(
            f"Sub-question {i + 1}: {pa['question']}\nAnswer: {pa['answer']}"
            for i, pa in enumerate(prior)
        )
        merge_user = (
            f"Original question: {state['question']}\n\n"
            f"{qa_block}\n\n"
            "Please combine these into a single coherent response."
        )

        try:
            llm = self.factory.get_llm()
            response = llm.invoke(
                [
                    SystemMessage(content=MERGE_PROMPT),
                    HumanMessage(content=merge_user),
                ]
            )
            merged = response.content if hasattr(response, "content") else str(response)
            return {**state, "final_answer": merged.strip()}
        except Exception as e:
            print(f"  [QAAgent] Merge failed ({e}), concatenating answers")
            fallback = "\n\n".join(
                f"{pa['question']}\n{pa['answer']}" for pa in prior
            )
            return {**state, "final_answer": fallback}

    def _node_think(self, state: AgentState) -> AgentState:
        """LLM decides: call a tool (TOOL:) or give the final answer (FINAL_ANSWER:)."""
        print("  [QAAgent] Thinking...")
        if state["steps_taken"] >= MAX_STEPS:
            sq = state["sub_questions"][state["current_index"]]
            print(f"  [QAAgent] Q{sq['id']}: max steps reached — forcing answer")
            summary = self._summarize_results(state["messages"])
            return {**state, "current_answer": f"Based on what I found:\n\n{summary}"}

        try:
            llm = self.factory.get_llm()
            response = llm.invoke(state["messages"])
            content = response.content if hasattr(response, "content") else str(response)
            sq = state["sub_questions"][state["current_index"]]
            preview = content[:100].replace("\n", " ").strip()
            print(f"  [QAAgent] Q{sq['id']} step {state['steps_taken'] + 1}: {preview}...")
            print(state["messages"] + [AIMessage(content=content)])
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content=content)],
            }
        except Exception as e:
            print(f"  [QAAgent] LLM error: {e}")
            return {**state, "current_answer": f"Error during analysis: {e}"}

    def _node_tool(self, state: AgentState) -> AgentState:
        """Parse TOOL/ARGS from last AI message, execute tool, append result."""
        last_content = self._last_ai_content(state["messages"])
        tool_name, tool_args = self._parse_tool_call(last_content)
        print(f"  [QAAgent] Parsed tool call: {tool_name} with args {tool_args}")
        sq = state["sub_questions"][state["current_index"]]
        print(f"  [QAAgent] Q{sq['id']} tool: {tool_name}({tool_args})")

        try:
            result = self._execute_tool(tool_name, tool_args)
        except Exception as e:
            result = {"error": f"Tool failed: {e}"}
            print(f"  [QAAgent] Tool error: {e}")

        result_text = (
            f"TOOL_RESULT [{tool_name}]:\n"
            f"{json.dumps(result, indent=2, default=str)}"
        )

        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content=result_text)],
            "steps_taken": state["steps_taken"] + 1,
        }

    def _node_finalize(self, state: AgentState) -> AgentState:
        """Extract FINAL_ANSWER from message history into current_answer."""
        print("  [QAAgent] Finalizing answer from message history...")
        if state.get("current_answer"):
            return state

        for msg in reversed(state["messages"]):
            content = msg.content if hasattr(msg, "content") else ""
            if "FINAL_ANSWER:" in content:
                answer = self._extract_final_answer(content)
                if answer is not None:
                    return {**state, "current_answer": answer}

        # Fallback: last AI message
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                answer_str = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                return {**state, "current_answer": answer_str}

        return {**state, "current_answer": "No answer generated."}

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — routing
    # ─────────────────────────────────────────────────────────────

    def _route_after_think(self, state: AgentState) -> str:
        print("  [QAAgent] Routing after think...")
        if state.get("current_answer"):
            print("  [QAAgent] 11111111/ Final answer found in state, routing to finalize")
            return "finalize"
        content = self._last_ai_content(state["messages"])
        if "FINAL_ANSWER:" in content:
            print("  [QAAgent] 22222222/ FINAL_ANSWER: found in content, routing to finalize and content is ", content)
            return "finalize"
        elif "TOOL:" in content:
            print("  [QAAgent] 33333333/ TOOL: found in content, routing to tool")
            return "tool"
        return "finalize"

    def _route_after_accumulate(self, state: AgentState) -> str:
        if state["current_index"] < len(state["sub_questions"]):
            return "next_question"
        return "merge"

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — tool execution
    # ─────────────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, args: dict) -> Any:
        t = self.tracer

        if tool_name == "error_frequency":
            print(f"  [QAAgent] Executing tool: error_frequency with args {args}")
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
                    f"root_cause, health_check, incident_duration, search, "
                    f"get_related_logs, get_entry_details"
                )
            }

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — helpers
    # ─────────────────────────────────────────────────────────────

    def _clean_json_output(self, raw: str) -> str:
        """
        Repair common small-model JSON mistakes before calling json.loads().

        Handles (in order):
          1. <think>...</think> blocks  — Qwen thinking mode emits these before output
          2. Markdown fences            — ```json ... ``` wrappers
          3. Prose before/after array   — extracts [...] even if surrounded by text
          4. Trailing commas            — ,] and ,} are invalid JSON
          5. Missing commas             — }\\n{ between objects needs a comma
          6. Python None                — must be JSON null
          7. Single-quoted strings      — must be double-quoted in JSON
        """
        # 1. Strip <think>...</think> reasoning blocks (Qwen thinking mode)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # 2. Strip markdown fences (regex avoids the lstrip character-stripping bug)
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw).strip()
        # 3. Extract the JSON array even if prose surrounds it
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        # 4. Remove trailing commas before ] or }
        raw = re.sub(r",\s*([\]}])", r"\1", raw)
        # 5. Insert missing commas between consecutive objects
        raw = re.sub(r"\}\s*\n\s*\{", "}, {", raw)
        # 6. Replace Python None with JSON null
        raw = re.sub(r":\s*None\b", ": null", raw)
        # 7. Replace single-quoted strings with double-quoted
        #    Only replaces 'value' patterns — avoids breaking already-valid JSON
        raw = re.sub(r"'([^']*)'", r'"\1"', raw)
        # 8. Fix array closed with } instead of ] — qwen2.5:3b emits this consistently:
        #    [ {...}, {...} }  instead of  [ {...}, {...} ]
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
                # Fallback: extract string-valued pairs only
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

        # Strip if the LLM leaked system prompt sections after the answer
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

        # Reject placeholder templates
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
        builder.add_node("merge_answers", self._node_merge_answers)

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
                "merge": "merge_answers",
            },
        )
        builder.add_edge("merge_answers", END)

        self._graph = builder.compile()
        return self._graph
