"""
root_cause_analyzer.py
=======================
Analyzes root cause of the last error cluster using LLM.
Provider-agnostic — uses LLMFactory (OpenAI / Anthropic / Ollama / Custom).

Usage:
    #from llm.llm_factory import LLMFactory
    from pylogtracer.llm.llm_factory import LLMFactory
    factory  = LLMFactory({"provider": "openai", "model": "gpt-4o"})
    analyzer = RootCauseAnalyzer(factory=factory)
    result   = analyzer.analyze(extraction_result)
"""

from typing import Dict, Optional

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# ANALYSIS_PROMPT built lazily inside _get_chain() to avoid import at module load


class RootCauseAnalyzer:
    """
    Sends the last error cluster to LLM and returns structured root cause analysis.

    Args:
        factory: LLMFactory instance.
    """

    def __init__(self, factory):
        self.factory = factory
        self._chain = None  # lazy-init LCEL chain

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def analyze(self, extraction_result: Dict, extra_context: Optional[str] = None) -> Dict:
        """
        Analyze root cause from ErrorExtractor output.

        Args:
            extraction_result: Output from ErrorExtractor.extract()
            extra_context:     Extra log lines fetched by ContextBridge (if any)

        Returns:
            {
                python_root_cause, root_cause, error_chain,
                suggested_fix, frequency_summary, raw_response
            }
        """
        last_cluster = extraction_result.get("last_cluster")
        frequency = extraction_result.get("frequency", {})

        if not last_cluster:
            return {"error": "No error cluster found."}

        python_guess = last_cluster[0]["primary_error"]
        context_text = self._build_context(last_cluster, extra_context)
        freq_summary = self._build_freq_summary(frequency)

        raw_response = self._call_llm(python_guess, freq_summary, context_text, extra_context)
        parsed = self._parse_response(raw_response)

        return {
            "python_root_cause": python_guess,
            "root_cause": parsed.get("root_cause", "See raw_response"),
            "error_chain": parsed.get("error_chain", ""),
            "suggested_fix": parsed.get("suggested_fix", ""),
            "frequency_summary": frequency,
            "raw_response": raw_response,
        }

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — LCEL chain
    # ─────────────────────────────────────────────────────────────

    def _get_chain(self):
        """Lazy-init: prompt | llm | str output parser."""
        if self._chain is None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert software engineer analyzing production log errors.\n"
                        "Analyze the error chain carefully. Some errors are side effects of an earlier failure.\n\n"
                        "DECISION RULE:\n"
                        "- If the error list clearly shows what went wrong -> give your best answer.\n"
                        "- ONLY ask for more context if errors are truly ambiguous AND more lines would change your answer.\n"
                        "- Do NOT ask for more context just because tracebacks are missing.\n\n"
                        "If you can identify the root cause respond ONLY in this exact format:\n"
                        "ROOT CAUSE: <single true root cause in one sentence>\n"
                        "ERROR CHAIN: <numbered steps showing how errors cascaded>\n"
                        "SUGGESTED FIX: <concrete actionable fix>\n\n"
                        "Only if truly ambiguous and more lines would change your answer:\n"
                        "NEED_MORE_CONTEXT: timestamp=<YYYY-MM-DD HH:MM:SS> | reason=<what is missing and why>",
                    ),
                    (
                        "human",
                        "Python pre-analysis identified this as the first error in the cluster:\n"
                        "  >> {python_guess}\n\n"
                        "Top recurring errors:\n{freq_summary}\n\n"
                        "Last error cluster:\n---\n{context_text}\n---\n"
                        "{extra_context_block}",
                    ),
                ]
            )
            self._chain = prompt | self.factory.get_llm() | StrOutputParser()
        return self._chain

    def _call_llm(self, python_guess, freq_summary, context_text, extra_context) -> str:
        extra_block = ""
        if extra_context:
            extra_block = f"\n[ADDITIONAL CONTEXT fetched on request]\n{extra_context}"
        try:
            return self._get_chain().invoke(
                {
                    "python_guess": python_guess,
                    "freq_summary": freq_summary,
                    "context_text": context_text,
                    "extra_context_block": extra_block,
                }
            )
        except Exception as e:
            return f"[LLM call failed]: {e}"

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — builders
    # ─────────────────────────────────────────────────────────────

    def _build_context(self, cluster: list, extra_context: Optional[str] = None) -> str:
        parts = []
        for i, err in enumerate(cluster, 1):
            ts = err["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if err["timestamp"] else "no-timestamp"
            parts.append(f"[Error {i}] {ts} | Type: {err['error_type']}")
            parts.append(f"  {err['primary_error']}")
            if err.get("traceback"):
                parts.append(err["traceback"])
            parts.append("")
        return "\n".join(parts)

    def _build_freq_summary(self, frequency: dict) -> str:
        if not frequency:
            return "  No frequency data"
        return "\n".join(f"  - {etype}: {count} time(s)" for etype, count in list(frequency.items())[:5])

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — response parser
    # ─────────────────────────────────────────────────────────────

    def _parse_response(self, response: str) -> Dict:
        result = {}
        current_key = None
        current_val = []

        key_map = {
            "ROOT CAUSE": "root_cause",
            "ERROR CHAIN": "error_chain",
            "SUGGESTED FIX": "suggested_fix",
        }

        for line in response.splitlines():
            matched = False
            for label, key in key_map.items():
                if line.startswith(f"{label}:"):
                    if current_key:
                        result[current_key] = "\n".join(current_val).strip()
                    current_key = key
                    current_val = [line[len(label) + 1 :].strip()]
                    matched = True
                    break
            if not matched and current_key:
                current_val.append(line)

        if current_key:
            result[current_key] = "\n".join(current_val).strip()

        return result
