"""
context_bridge.py
==================
Agent-to-agent communication layer.

When RootCauseAnalyzer calls Qwen and Qwen says
"I need more context around timestamp X", ContextBridge:

  1. Detects the NEED_MORE_CONTEXT signal in Qwen's response
  2. Calls SmartReader.fetch_lines_around(timestamp, context)
  3. Feeds the extra lines back to RootCauseAnalyzer for a retry
  4. Repeats up to MAX_RETRIES times
  5. Returns the final answer

This means the LLM can ask for more information without you
manually re-running anything — the agents talk to each other.

Usage:
    bridge = ContextBridge(reader=reader, analyzer=analyzer, max_retries=2)
    result = bridge.run(extraction_result)
"""

import re
from typing import Dict, Optional


# How Qwen signals it needs more context.
# ContextBridge scans every Qwen response for this pattern.
#
# The LLM should respond with:
#   NEED_MORE_CONTEXT: timestamp=2024-03-01 10:00:05 | reason=traceback incomplete
#
NEED_MORE_CONTEXT_PATTERN = re.compile(
    r"NEED_MORE_CONTEXT:\s*timestamp=([^\|]+)\|?\s*(?:reason=(.+))?",
    re.IGNORECASE,
)

# Max times we allow Qwen to ask for more context before we stop.
# Prevents infinite loops on badly-behaved models.
MAX_RETRIES = 3


class ContextBridge:
    """
    Sits between RootCauseAnalyzer and SmartReader.
    Handles agent-to-agent back-and-forth automatically.

    Args:
        reader:      Instance of get_file_content (smart_reader).
                     Must have already called fetch_logs_by_date() so
                     _all_lines cache is populated.
        analyzer:    Instance of RootCauseAnalyzer.
        max_retries: Max times Qwen can request more context. Default 3.
        context_lines_per_request: Lines before+after timestamp to fetch. Default 15.
    """

    def __init__(
        self,
        reader,
        analyzer,
        max_retries:               int = MAX_RETRIES,
        context_lines_per_request: int = 15,
    ):
        self.reader                    = reader
        self.analyzer                  = analyzer
        self.max_retries               = max_retries
        self.context_lines_per_request = context_lines_per_request

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def run(self, extraction_result: Dict) -> Dict:
        """
        Run the full agent loop.

        1. Calls RootCauseAnalyzer with the extraction result.
        2. If the LLM asks for more context → fetches it from SmartReader.
        3. Retries with the extra context appended.
        4. Returns when Qwen gives a final answer or max_retries hit.

        Args:
            extraction_result: Output from ErrorExtractor.extract()

        Returns:
            Final analysis dict from RootCauseAnalyzer, plus:
            {
                ...normal analyzer output...,
                "retries_used":   int,
                "extra_contexts": list of timestamps fetched
            }
        """
        retries_used   = 0
        extra_contexts = []          # accumulates extra lines across retries
        extra_text     = ""          # injected into analyzer on each retry

        while retries_used <= self.max_retries:

            # ── Call analyzer (with any extra context appended) ───
            result = self.analyzer.analyze(
                extraction_result,
                extra_context=extra_text or None,
            )

            raw_response = result.get("raw_response", "")

            # ── Check if Qwen is asking for more context ──────────
            need_more = self._detect_need_more_context(raw_response)

            if need_more is None:
                # Qwen gave a final answer — we're done
                result["retries_used"]   = retries_used
                result["extra_contexts"] = extra_contexts
                return result

            # ── Qwen wants more — ask SmartReader ─────────────────
            timestamp = need_more["timestamp"]
            reason    = need_more["reason"]

            print(f"\n  [ContextBridge] Qwen needs more context:")
            print(f"    Timestamp : {timestamp}")
            print(f"    Reason    : {reason or 'not specified'}")
            print(f"    Fetching  : ±{self.context_lines_per_request} lines from SmartReader...")

            fetch_result = self.reader.fetch_lines_around(
                timestamp     = timestamp,
                context_lines = self.context_lines_per_request,
            )

            if not fetch_result.get("found"):
                print(f"  [ContextBridge] Timestamp not found in file — stopping retry.")
                result["retries_used"]   = retries_used
                result["extra_contexts"] = extra_contexts
                return result

            # Append new context to accumulate across retries
            extra_contexts.append(timestamp)
            extra_text += f"\n\n[Extra context around {timestamp} — fetched on request]\n"
            extra_text += fetch_result["lines"]

            print(f"  [ContextBridge] Got {len(fetch_result['lines'].splitlines())} extra lines. Retrying...")

            retries_used += 1

        # Max retries hit — return whatever we have
        print(f"\n  [ContextBridge] Max retries ({self.max_retries}) reached.")
        result["retries_used"]   = retries_used
        result["extra_contexts"] = extra_contexts
        return result

    # ─────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────

    def _detect_need_more_context(self, response: str) -> Optional[Dict]:
        """
        Scan Qwen's response for the NEED_MORE_CONTEXT signal.

        Returns dict with timestamp + reason if found, else None.
        """
        match = NEED_MORE_CONTEXT_PATTERN.search(response)
        if not match:
            return None

        return {
            "timestamp": match.group(1).strip(),
            "reason":    (match.group(2) or "").strip(),
        }
