"""
logtracer.py
=============
Public API for the log analyzer package.
All internal modules are hidden — user only interacts with LogTracer.

Two modes:
  1. Library mode  — direct function calls, no agent needed
  2. Agent mode    — logtracer.ask() uses LangGraph to answer free-form questions

Usage:
    from logtracer import LogTracer

    # Init once
    tracer = LogTracer(
        file_path  = "app.log",
        llm_config = {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-..."}
        # or leave llm_config=None to read from .env
    )

    # Library mode — direct calls
    tracer.error_frequency()
    tracer.summary()
    tracer.errors_by_date("2024-03-01")
    tracer.errors_in_range("2024-03-01 09:00:00", "2024-03-01 11:00:00")
    tracer.root_cause_analysis()

    # Agent mode — free-form questions
    tracer.ask("what caused the crash at 10am?")
    tracer.ask("how many DB errors happened today?")
    tracer.ask("show me errors between 9am and 11am")
"""

import logging
from re import S
from typing import Optional, Dict, List, Any

# Setup logging for tool calls
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from pylogtracer.preprocessing.smart_reader import get_file_content
from pylogtracer.preprocessing.error_extractor import ErrorExtractor
from pylogtracer.preprocessing.error_type_classifier import ErrorTypeClassifier
from pylogtracer.agents.root_cause_analyzer import RootCauseAnalyzer
from pylogtracer.multiagent.context_bridge import ContextBridge
from pylogtracer.llm.llm_factory import LLMFactory


class LogTracer:
    """
    Main public API for log analysis.

    Args:
        file_path:   Path to the log file to analyze
        llm_config:  LLM provider config dict. None = reads from .env
                     Keys: provider, model, api_key, base_url, temperature, max_tokens
        gap_seconds: Time gap (seconds) to separate error incidents. Default 60.
        max_retries: Max times LLM can request more context. Default 2.
    """

    def __init__(
        self,
        file_path: str,
        llm_config: Optional[Dict] = None,
        gap_seconds: int = 60,
        max_retries: int = 2,
    ):
        self.file_path = file_path
        self.gap_seconds = gap_seconds
        self.max_retries = max_retries

        # LLM factory — shared across all modules
        self._factory = LLMFactory(llm_config)

        # Internal state — lazily populated
        self._reader: Any = None
        self._extraction: Optional[Dict[str, Any]] = None  # cached extraction result
        self._last_filter: Optional[  # track which filter was used for cache
            tuple[Optional[str], Optional[str], Optional[str]]
        ] = None

        # Persist classifier across ask() calls so keyword store survives
        # between questions — avoids re-learning same keywords every call
        self._classifier = ErrorTypeClassifier(factory=self._factory)

    # ─────────────────────────────────────────────────────────────
    # PUBLIC — Library mode
    # ─────────────────────────────────────────────────────────────

    def error_frequency(
        self,
        date: Optional[str] = None,
        from_dt: Optional[str] = None,
        to_dt: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Count how many times each error type occurred.

        Args:
            date:    Filter by specific date e.g. "2024-03-01"
            from_dt: Range start e.g. "2024-03-01 09:00:00"
            to_dt:   Range end   e.g. "2024-03-01 11:00:00"

        Returns:
            { "DatabaseConnectionError": 4, "ZeroDivisionError": 1, ... }

        Example:
            tracer.error_frequency()
            tracer.error_frequency(date="2024-03-01")
        """
        extraction = self._get_extraction(date=date, from_dt=from_dt, to_dt=to_dt)
        return extraction["frequency"]

    def summary(
        self,
        date: Optional[str] = None,
        from_dt: Optional[str] = None,
        to_dt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        High-level summary of the log file or a filtered time range.

        Returns:
            {
                "total_entries"  : int,
                "total_errors"   : int,
                "total_clusters" : int,
                "error_types"    : list of unique error types,
                "first_error"    : timestamp of first error,
                "last_error"     : timestamp of last error,
                "frequency"      : { error_type: count },
                "filter"         : what filter was applied
            }

        Example:
            tracer.summary()
            tracer.summary(date="2024-03-01")
        """
        read_result = self._read(date=date, from_dt=from_dt, to_dt=to_dt)
        extraction = self._get_extraction(date=date, from_dt=from_dt, to_dt=to_dt)

        all_errors = extraction["all_errors"]
        timestamps = [e["timestamp"] for e in all_errors if e["timestamp"]]

        return {
            "total_entries": read_result["total_matched"],
            "total_errors": extraction["total_errors"],
            "total_clusters": len(extraction["clusters"]),
            "error_types": list(extraction["frequency"].keys()),
            "first_error": min(timestamps).strftime("%Y-%m-%d %H:%M:%S") if timestamps else None,
            "last_error": max(timestamps).strftime("%Y-%m-%d %H:%M:%S") if timestamps else None,
            "frequency": extraction["frequency"],
            "filter": read_result["filter"],
        }

    def errors_by_date(self, date: str) -> List[Dict]:
        """
        Get all errors for a specific date.

        Args:
            date: "YYYY-MM-DD" format

        Returns:
            List of parsed error dicts for that date.

        Example:
            tracer.errors_by_date("2024-03-01")
        """
        extraction = self._get_extraction(date=date)
        return extraction["all_errors"]

    def errors_in_range(self, from_dt: str, to_dt: str) -> List[Dict]:
        """
        Get all errors between two timestamps.

        Args:
            from_dt: "YYYY-MM-DD HH:MM:SS"
            to_dt:   "YYYY-MM-DD HH:MM:SS"

        Returns:
            List of parsed error dicts in that range.

        Example:
            tracer.errors_in_range("2024-03-01 09:00:00", "2024-03-01 11:00:00")
        """
        extraction = self._get_extraction(from_dt=from_dt, to_dt=to_dt)
        return extraction["all_errors"]

    def root_cause_analysis(
        self,
        date: Optional[str] = None,
        from_dt: Optional[str] = None,
        to_dt: Optional[str] = None,
    ) -> Dict:
        """
        Analyze root cause of the last error cluster using LLM.
        LLM will automatically request more context lines if needed.

        Returns:
            {
                "root_cause"    : str,
                "error_chain"   : str,
                "suggested_fix" : str,
                "frequency"     : { error_type: count },
                "retries_used"  : int
            }

        Example:
            tracer.root_cause_analysis()
            tracer.root_cause_analysis(date="2024-03-01")
        """
        extraction = self._get_extraction(date=date, from_dt=from_dt, to_dt=to_dt)

        if not extraction["all_errors"]:
            return {"error": "No errors found in the specified range."}

        reader = self._get_reader(date=date, from_dt=from_dt, to_dt=to_dt)
        analyzer = RootCauseAnalyzer(factory=self._factory)
        bridge = ContextBridge(reader=reader, analyzer=analyzer, max_retries=self.max_retries)
        return bridge.run(extraction)

    def last_incident(
        self,
        date: Optional[str] = None,
        from_dt: Optional[str] = None,
        to_dt: Optional[str] = None,
    ) -> List[Dict]:
        """
        Return the most recent error cluster (last incident).

        Example:
            tracer.last_incident()
        """
        extraction = self._get_extraction(date=date, from_dt=from_dt, to_dt=to_dt)
        return extraction["last_cluster"] or []

    def search(self, keyword: str, max_results: int = 20) -> Dict:
        """
        Search logs for any keyword or unique identifier.
        Returns most recent matches first.

        Args:
            keyword:     Any string — "INC1033234", "connection refused", "db:5432"
            max_results: Max results to return. Default 20.

        Returns:
            {
                "keyword":     str,
                "total_found": int,
                "entries":     list of matched log entries (recent first)
            }

        Example:
            tracer.search("INC1033234")
            tracer.search("connection refused")
        """
        msg = f"[SEARCH] Searching logs for keyword '{keyword}' with max_results={max_results}..."
        print(msg)
        logger.info(msg)
        reader = self._get_reader()
        return reader.search_logs(keyword, max_results=max_results)

    def health_check(self) -> Dict:
        """
        Check if the system is healthy based on recent log activity.

        Returns:
            {
                "healthy":        bool,
                "status":         str,    "OK" | "WARNING" | "CRITICAL"
                "total_errors":   int,
                "last_error":     str | None,
                "last_error_type": str | None,
                "summary":        str
            }

        Example:
            tracer.health_check()
        """
        extraction = self._get_extraction()
        total = extraction["total_errors"]
        last = extraction["last_cluster"]

        if total == 0:
            return {
                "healthy": True,
                "status": "OK",
                "total_errors": 0,
                "last_error": None,
                "last_error_type": None,
                "summary": "No errors found. System appears healthy.",
            }

        last_error = last[-1] if last else None
        last_ts = last_error["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if last_error and last_error["timestamp"] else None
        last_error_type = last_error["error_type"] if last_error else None

        # CRITICAL if any CRITICAL/FATAL in last cluster
        has_critical = any(
            "critical" in e["primary_error"].lower() or "fatal" in e["primary_error"].lower() for e in (last or [])
        )

        status = "CRITICAL" if has_critical else "WARNING"

        return {
            "healthy": False,
            "status": status,
            "total_errors": total,
            "last_error": last_ts,
            "last_error_type": last_error_type,
            "summary": f"{status}: {total} error(s) found. Last error: {last_error_type} at {last_ts}.",
        }

    def incident_duration(
        self,
        date: Optional[str] = None,
        from_dt: Optional[str] = None,
        to_dt: Optional[str] = None,
    ) -> Dict:
        """
        Calculate how long the last incident lasted.

        Returns:
            {
                "start":           str,
                "end":             str,
                "duration_seconds": int,
                "duration_human":  str,   "2 minutes 6 seconds"
                "error_count":     int
            }

        Example:
            tracer.incident_duration()
        """
        extraction = self._get_extraction(date=date, from_dt=from_dt, to_dt=to_dt)
        last = extraction.get("last_cluster") or []

        if not last:
            return {"error": "No incident found."}

        timestamps = [e["timestamp"] for e in last if e["timestamp"]]
        if not timestamps:
            return {"error": "No timestamps in last cluster."}

        start = min(timestamps)
        end = max(timestamps)
        duration = int((end - start).total_seconds())

        # Human readable
        mins, secs = divmod(duration, 60)
        hours, mins = divmod(mins, 60)
        if hours:
            human = f"{hours}h {mins}m {secs}s"
        elif mins:
            human = f"{mins} minute(s) {secs} second(s)"
        else:
            human = f"{secs} second(s)"

        return {
            "start": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end": end.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "duration_human": human,
            "error_count": len(last),
        }

    def get_related_logs(self, identifier: str) -> Dict:
        """
        Find all logs related to an identifier — both error cluster entries
        and non-error entries (INFO, DEBUG, WARNING) that mention the same ID.

        Strategy (in order):
          1. search() to find ALL log lines that mention the identifier.
             This is the superset — guaranteed to find everything.
          2. Among those matches, scan error clusters for any entry whose
             full_entry or primary_error contains the identifier.
             If a cluster match is found → include the full cluster too
             (gives error-scoped analysis alongside the raw lines).
          3. Return both sets clearly labelled so the LLM has full picture.

        Why this is better than the old approach:
          OLD: took entries[0] (most recent raw string) as anchor, then tried
               to match its first 80 chars against full_entry in clusters.
               Failed silently when the most recent match was an INFO line
               (never in any cluster) or when whitespace differed.
          NEW: searches clusters by identifier string directly — same lookup
               that found the raw entries, so it never misses.

        Args:
            identifier: Any string present in log entries
                        e.g. "INC1033234", "connection refused", "REQ-456"

        Returns:
            {
                "identifier":        str,
                "found":             bool,
                "total_found":       int,     all lines mentioning identifier
                "all_entries":       list,    every matching raw log line
                "error_cluster":     list,    entries from error cluster (may be empty)
                "cluster_index":     int|None,
                "total_in_cluster":  int,
                "has_error_cluster": bool,    True if identifier is in an error cluster
                "note":              str      human-readable explanation of what was found
            }

        Example:
            tracer.get_related_logs("INC1033234")
            tracer.get_related_logs("connection refused")
        """
        print(f"  [get_related_logs] Searching for '{identifier}'...")

        # ── Step 1: search ALL log lines for the identifier ───────────────
        # Increase max_results so we don't miss any entries for busy incidents
        search_result = self.search(identifier, max_results=50)
        all_entries = search_result.get("entries", [])
        total_found = search_result.get("total_found", 0)

        if not all_entries:
            print(f"  [get_related_logs] No entries found for '{identifier}'")
            return {
                "identifier": identifier,
                "found": False,
                "total_found": 0,
                "all_entries": [],
                "error_cluster": [],
                "cluster_index": None,
                "total_in_cluster": 0,
                "has_error_cluster": False,
                "note": f"No log entries found containing '{identifier}'.",
            }

        print(f"  [get_related_logs] Found {total_found} raw entries. Scanning error clusters...")

        # ── Step 2: scan error clusters for the identifier ────────────────
        # Match directly on identifier string — avoids the fragile 80-char
        # fingerprint approach and works regardless of entry format.
        extraction = self._get_extraction()
        clusters = extraction.get("clusters", [])

        identifier_lower = identifier.lower()
        matched_cluster = None
        matched_cluster_index = None

        for ci, cluster in enumerate(clusters):
            for error in cluster:
                # Check both full_entry and primary_error — either can carry the ID
                full = error.get("full_entry", "").lower()
                primary = error.get("primary_error", "").lower()
                if identifier_lower in full or identifier_lower in primary:
                    matched_cluster = cluster
                    matched_cluster_index = ci
                    break
            if matched_cluster is not None:
                break

        # ── Step 3: format the cluster entries if found ───────────────────
        cluster_formatted = []
        if matched_cluster:
            cluster_formatted = [
                {
                    "timestamp": (
                        e["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                        if e.get("timestamp") else None
                    ),
                    "error_type": e.get("error_type"),
                    "primary_error": e.get("primary_error"),
                    "traceback": e.get("traceback", ""),
                    "full_entry": e.get("full_entry", ""),
                }
                for e in matched_cluster
            ]
            note = (
                f"Found {total_found} log entries and {len(cluster_formatted)} "
                f"entries in error cluster {matched_cluster_index}."
            )
            print(f"  [get_related_logs] Matched error cluster {matched_cluster_index} "
                  f"({len(cluster_formatted)} entries)")
        else:
            note = (
                f"Found {total_found} log entries for '{identifier}'. "
                f"None belong to an error cluster (likely INFO/DEBUG/WARNING lines only)."
            )
            print(f"  [get_related_logs] No error cluster match — returning raw entries only")

        return {
            "identifier": identifier,
            "found": True,
            "total_found": total_found,
            "all_entries": all_entries,          # ALL lines — INFO, ERROR, DEBUG etc.
            "error_cluster": cluster_formatted,  # only the error-cluster subset (may be [])
            "cluster_index": matched_cluster_index,
            "total_in_cluster": len(cluster_formatted),
            "has_error_cluster": matched_cluster is not None,
            "note": note,
        }

    def get_entry_details(self, identifier: str) -> Dict:
        """
        Get full details of a log entry matching an identifier.
        Returns most recent match first.

        Args:
            identifier: Any unique string in the log entry
                        e.g. "INC1033234", timestamp, error message snippet

        Returns:
            {
                "identifier": str,
                "found":      bool,
                "entries":    list of full entry dicts with traceback
            }

        Example:
            tracer.get_entry_details("INC1033234")
            tracer.get_entry_details("10:00:05")
        """
        search_result = self.search(identifier)
        if not search_result.get("entries"):
            return {
                "identifier": identifier,
                "found": False,
                "entries": [],
            }

        # Parse each matched entry for full details
        from pylogtracer.preprocessing.error_extractor import ErrorExtractor

        extractor = ErrorExtractor()
        entries = []
        for raw_entry in search_result["entries"]:
            parsed = extractor._parse_error_entry(raw_entry)
            parsed["raw"] = raw_entry
            if parsed.get("timestamp"):
                parsed["timestamp"] = parsed["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            entries.append(parsed)

        return {
            "identifier": identifier,
            "found": True,
            "entries": entries,
        }

    def ask(self, question: str) -> str:
        """
        Ask a free-form question about the logs.
        Uses LangGraph agent to decide which tools to call.

        Args:
            question: Natural language question about the logs

        Returns:
            str: Answer to the question

        Example:
            tracer.ask("what caused the crash at 10am?")
            tracer.ask("how many DB errors happened today?")
            tracer.ask("show me all errors between 9am and 11am")
        """
        from pylogtracer.agents.qa_agent import QAAgent

        agent = QAAgent(tracer=self, factory=self._factory)
        return agent.run(question)

    # ─────────────────────────────────────────────────────────────
    # INTERNAL — used by qa_agent.py tools (not for direct user call)
    # ─────────────────────────────────────────────────────────────

    def _read(
        self,
        date: Optional[str] = None,
        from_dt: Optional[str] = None,
        to_dt: Optional[str] = None,
        relative_day: Optional[str] = None,
    ) -> Dict:
        """Internal: run SmartReader and return raw read result."""
        reader = get_file_content(
            relative_day=relative_day,
            date=date,
            from_dt=from_dt,
            to_dt=to_dt,
        )
        result = reader.fetch_logs_by_date(self.file_path)
        if "error" in result:
            raise RuntimeError(result["error"])
        # Cache reader for context_bridge use
        self._reader = reader
        return result

    def _get_reader(
        self,
        date: Optional[str] = None,
        from_dt: Optional[str] = None,
        to_dt: Optional[str] = None,
    ):
        """Internal: return cached reader or create new one."""
        if self._reader is None:
            self._read(date=date, from_dt=from_dt, to_dt=to_dt)
        self._read(date=date, from_dt=from_dt, to_dt=to_dt)
        logger.debug(f"SmartReader instance ready for use: {self._reader}")
        return self._reader

    def _get_extraction(
        self,
        date: Optional[str] = None,
        from_dt: Optional[str] = None,
        to_dt: Optional[str] = None,
    ) -> Dict:
        """
        Internal: run full extraction pipeline.
        Caches result — same filter won't re-run extraction.
        """
        current_filter = (date, from_dt, to_dt)

        # Return cache if same filter
        if self._extraction and self._last_filter == current_filter:
            return self._extraction

        # Read logs
        read_result = self._read(date=date, from_dt=from_dt, to_dt=to_dt)
        log_entries = read_result["logs"]

        # Extract + classify — reuse persisted classifier so keyword
        # store survives across multiple ask() calls this session
        extractor = ErrorExtractor(
            gap_seconds=self.gap_seconds,
            classifier=self._classifier,
        )
        self._extraction = extractor.extract(log_entries)
        self._last_filter = current_filter

        return self._extraction