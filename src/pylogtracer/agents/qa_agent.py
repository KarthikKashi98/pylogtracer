"""
qa_agent.py
============
Dynamic ReAct agent for LogTracer.ask()

Architecture — LangGraph ReAct loop:

    [START]
       ↓
    [think]  ← LLM decides: which tool? or am I done?
       ↓  ↑
    [tool]   ← executes tool, result appended to message history
       ↓  ↑______ loop back if LLM wants more tools
    [finalize] ← extract final answer
       ↓
    [END]

Key difference from old static router:
  OLD: question → pick ONE tool → format answer (1 step, always)
  NEW: question → think → tool → think → tool → ... → answer (N steps)

This means:
  "show INC1033234 and how long it lasted"
    Step 1: get_related_logs("INC1033234") → sees cluster
    Step 2: incident_duration()            → sees duration
    Step 3: FINAL_ANSWER combining both

  "compare errors today vs yesterday"
    Step 1: errors_by_date(today)     → today's errors
    Step 2: errors_by_date(yesterday) → yesterday's errors
    Step 3: FINAL_ANSWER with comparison

  "is it getting worse?"
    Step 1: error_frequency()  → total counts
    Step 2: last_incident()    → recent cluster
    Step 3: FINAL_ANSWER with trend analysis

Max steps: 8 (prevents infinite loops)

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

MAX_STEPS = 8   # max tool calls per question


# ── Agent state ───────────────────────────────────────────────────
class AgentState(TypedDict):
    question:     str
    messages:     List[Any]
    steps_taken:  int
    final_answer: Optional[str]


# ── System prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a powerful log analysis agent.
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
  FINAL_ANSWER: <write your answer here — be specific with error types, timestamps, counts>

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
  - FINAL_ANSWER must contain ONLY your answer to the user's question, nothing else
"""


class QAAgent:
    """
    Dynamic ReAct agent — thinks, calls tools, loops until it can answer.

    Args:
        tracer:  LogTracer instance (all tools come from here)
        factory: LLMFactory instance (provides the LLM)
    """

    def __init__(self, tracer, factory):
        self.tracer  = tracer
        self.factory = factory
        self._graph  = None

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def run(self, question: str) -> str:
        """
        Answer any free-form question about the logs.
        Automatically resolves relative time references.
        Calls as many tools as needed to fully answer.
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph not installed. "
                "Run: pip install langgraph langchain-core"
            )

        # Resolve relative time before passing to agent
        from pylogtracer.utils.time_resolver import TimeResolver
        resolved = TimeResolver().resolve(question)
        enriched = resolved["enriched_question"]

        if resolved["resolved"]:
            print(f"  [QAAgent] Time resolved: "
                  f"from={resolved['from_dt']} | "
                  f"to={resolved['to_dt']} | "
                  f"date={resolved['date']}")

        graph = self._get_graph()
        state = graph.invoke({
            "question":     enriched,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=enriched),
            ],
            "steps_taken":  0,
            "final_answer": None,
        })

        return state["final_answer"] or "I could not find an answer."

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — LangGraph nodes
    # ─────────────────────────────────────────────────────────────

    def _node_think(self, state: AgentState) -> AgentState:
        """
        Think node — LLM sees full conversation history and decides:
          TOOL: ... → call a tool
          FINAL_ANSWER: ... → done
        """
        if state["steps_taken"] >= MAX_STEPS:
            print(f"  [QAAgent] Max steps reached — forcing final answer")
            summary = self._summarize_results(state["messages"])
            return {**state, "final_answer": f"Based on what I found:\n\n{summary}"}

        try:
            llm      = self.factory.get_llm()
            response = llm.invoke(state["messages"])
            content  = response.content if hasattr(response, "content") else str(response)

            step = state["steps_taken"] + 1
            preview = content[:100].replace("\n", " ").strip()
            print(f"  [QAAgent] Step {step}: {preview}...")

            return {
                **state,
                "messages": state["messages"] + [AIMessage(content=content)],
            }

        except Exception as e:
            print(f"  [QAAgent] LLM error: {e}")
            return {**state, "final_answer": f"Error during analysis: {e}"}

    def _node_tool(self, state: AgentState) -> AgentState:
        """
        Tool node — parse last AI message for TOOL/ARGS,
        execute tool, append result to message history.
        """
        last_content = self._last_ai_content(state["messages"])
        tool_name, tool_args = self._parse_tool_call(last_content)

        print(f"  [QAAgent] Tool: {tool_name}({tool_args})")

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
            "messages":   state["messages"] + [HumanMessage(content=result_text)],
            "steps_taken": state["steps_taken"] + 1,
        }

    def _node_check(self, state: AgentState) -> str:
        """
        Conditional edge — check last AI message:
          FINAL_ANSWER → finalize
          TOOL         → tool node
          else         → finalize (safety)
        """
        if state.get("final_answer"):
            return "end"

        content = self._last_ai_content(state["messages"])

        if "FINAL_ANSWER:" in content:
            return "end"
        elif "TOOL:" in content:
            return "tool"
        else:
            return "end"

    def _node_finalize(self, state: AgentState) -> AgentState:
        """Extract FINAL_ANSWER from message history."""
        if state.get("final_answer"):
            return state

        for msg in reversed(state["messages"]):
            content = msg.content if hasattr(msg, "content") else ""
            if "FINAL_ANSWER:" in content:
                answer = self._extract_final_answer(content)
                return {**state, "final_answer": answer}

        # Fallback to last AI message
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                return {**state, "final_answer": msg.content}

        return {**state, "final_answer": "No answer generated."}

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
            to_dt   = args.get("to_dt")
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
            return {"error": f"Unknown tool: {tool_name}. Available: error_frequency, errors_by_date, errors_in_range, last_incident, summary, root_cause, health_check, incident_duration, search, get_related_logs, get_entry_details"}

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — helpers
    # ─────────────────────────────────────────────────────────────

    def _parse_tool_call(self, content: str):
        """Parse TOOL name and ARGS dict from LLM response."""
        tool_m = re.search(r'TOOL:\s*(\w+)', content)
        args_m = re.search(r'ARGS:\s*(\{.*?\})', content, re.DOTALL)

        if not tool_m:
            return None, {}

        tool_name = tool_m.group(1).strip()
        args      = {}

        if args_m:
            try:
                raw  = args_m.group(1).strip()
                args = json.loads(raw)
                args = {k: v for k, v in args.items()
                        if v is not None and v != "null" and v != ""}
            except json.JSONDecodeError:
                # Manual extraction fallback
                pairs = re.findall(r'"(\w+)":\s*"([^"]+)"', args_m.group(1))
                args  = dict(pairs)

        return tool_name, args

    def _extract_final_answer(self, content: str) -> str:
        """Extract text after FINAL_ANSWER: — strips any leaked system prompt."""
        m = re.search(r'FINAL_ANSWER:\s*(.+)', content, re.DOTALL)
        if not m:
            return content.strip()

        answer = m.group(1).strip()

        # Strip leaked system prompt rules (qwen3 sometimes repeats them)
        cutoffs = [
            "\nyour complete answer here",
            "\nIMPORTANT RULES:",
            "\nTOOL:",
            "\nREACT LOOP",
            "\nTOOLS AVAILABLE:",
            "\n- Call tools ONE at a time",
            "\n- After each tool result",
            "\n- For multi-part questions",
            "\n- Use RESOLVED timestamps",
            "\n- Never invent data",
            "\n- Be specific",
        ]
        for cutoff in cutoffs:
            idx = answer.find(cutoff)
            if idx != -1:
                answer = answer[:idx].strip()

        return answer

    def _last_ai_content(self, messages: list) -> str:
        """Get content of last AIMessage."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content if hasattr(msg, "content") else str(msg)
        return ""

    def _safe_args(self, args: dict, allowed: list) -> dict:
        """Filter args to only allowed keys with non-empty values."""
        return {k: v for k, v in args.items()
                if k in allowed and v and v != "null"}

    def _fmt_errors(self, errors: list) -> list:
        """Serialize error list — convert datetimes to strings."""
        out = []
        for e in errors:
            out.append({
                "timestamp":     e["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                                 if e.get("timestamp") else None,
                "error_type":    e.get("error_type"),
                "primary_error": e.get("primary_error"),
                "traceback":     e.get("traceback", ""),
            })
        return out

    def _summarize_results(self, messages: list) -> str:
        """Collect all TOOL_RESULT blocks from history."""
        blocks = [
            msg.content for msg in messages
            if hasattr(msg, "content") and "TOOL_RESULT" in msg.content
        ]
        return "\n\n".join(blocks) if blocks else "No results collected."

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — LangGraph graph
    # ─────────────────────────────────────────────────────────────

    def _get_graph(self):
        """Build and cache the ReAct LangGraph."""
        if self._graph is not None:
            return self._graph

        builder = StateGraph(AgentState)

        builder.add_node("think",    self._node_think)
        builder.add_node("tool",     self._node_tool)
        builder.add_node("finalize", self._node_finalize)

        builder.add_edge(START,   "think")

        # think → tool (if TOOL found) OR finalize (if FINAL_ANSWER)
        builder.add_conditional_edges(
            "think",
            self._node_check,
            {"tool": "tool", "end": "finalize"}
        )

        # tool → think (ReAct loop)
        builder.add_edge("tool",     "think")
        builder.add_edge("finalize", END)

        self._graph = builder.compile()
        return self._graph