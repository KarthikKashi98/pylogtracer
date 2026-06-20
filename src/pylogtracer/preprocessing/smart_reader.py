"""
smart_reader.py
================
Reads a log file and filters lines by date/range.
Returns raw log string → passed directly to ErrorExtractor.

Also supports agent-to-agent communication:
    fetch_lines_around() lets RootCauseAnalyzer request
    more context around a specific timestamp when Qwen
    says it needs more information.

Usage:
    reader = get_file_content(relative_day="today")
    result = reader.fetch_logs_by_date("/var/log/app.log")
    raw_logs = result["logs"]   # ← pass this to ErrorExtractor

    # Agent callback — called by ContextBridge when Qwen needs more:
    extra = reader.fetch_lines_around(
        timestamp="2024-03-01 10:00:05",
        context_lines=10
    )
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from pylogtracer.preprocessing import log_format

logger = logging.getLogger(__name__)


class get_file_content:
    def __init__(
        self,
        relative_day=None,
        date=None,
        from_dt=None,
        to_dt=None,
        max_lines=None,
        max_bytes=None,
        tail=False,
        log_format_mode="auto",
        json_keys=None,
        glob_rotated=False,
        log_pattern=None,
        timestamp_format=None,
    ):
        if from_dt and not to_dt:
            to_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.relative_day = relative_day
        self.date = date
        self.from_dt = from_dt
        self.to_dt = to_dt
        # Bounded-read / format options (all default to today's behavior).
        # `tail` is a convenience alias for "keep only the recent window".
        if tail and not max_lines and not max_bytes:
            max_lines = 50_000
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.log_format_mode = log_format_mode
        self.json_keys = json_keys
        self.glob_rotated = glob_rotated
        self.log_pattern = log_pattern
        self.timestamp_format = timestamp_format
        self._capped = bool(max_lines or max_bytes)
        self._last_file = None  # set when fetch_logs_by_date() is called
        self._all_lines = []  # cached for fetch_lines_around() agent callback

    def _pick_datetime(self):
        now = datetime.now()

        if self.relative_day:
            rd = self.relative_day.lower()
            if rd == "today":
                return str(now.date())
            elif rd == "yesterday":
                return str((now - timedelta(days=1)).date())
            elif rd in ["tomorrow", "next day"]:
                return str((now + timedelta(days=1)).date())
            return "Invalid relative_day"

        elif self.date:
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(self.date, fmt).strftime("%Y-%m-%d")
                except ValueError:
                    pass
            return "Invalid datetime format"

        elif self.from_dt and self.to_dt:
            range_formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
                "%d-%m-%Y %H:%M:%S",
                "%d-%m-%Y",
            ]
            f = t = None
            for fmt in range_formats:
                try:
                    f = datetime.strptime(self.from_dt, fmt)
                    break
                except ValueError:
                    pass
            for fmt in range_formats:
                try:
                    t = datetime.strptime(self.to_dt, fmt)
                    break
                except ValueError:
                    pass

            if f is None:
                return "Invalid from_dt format"
            if t is None:
                return "Invalid to_dt format"
            if len(self.to_dt) == 10:
                t = t.replace(hour=23, minute=59, second=59)
            if f > t:
                return "from_dt cannot be greater than to_dt"
            return {
                "from": f.strftime("%Y-%m-%d %H:%M:%S"),
                "to": t.strftime("%Y-%m-%d %H:%M:%S"),
            }
        return None

    def fetch_logs_by_date(self, file_path: str) -> dict:
        """
        Read file and filter lines by date.

        Returns:
            {
                "file":          str,
                "filter":        str | dict | None,
                "total_matched": int,
                "logs":          str   ← pass this to ErrorExtractor
            }
        """
        try:
            lines = log_format.read_lines(
                file_path,
                max_lines=self.max_lines,
                max_bytes=self.max_bytes,
                log_format=self.log_format_mode,
                json_keys=self.json_keys,
                glob_rotated=self.glob_rotated,
                log_pattern=self.log_pattern,
                timestamp_format=self.timestamp_format,
            )
        except FileNotFoundError:
            return {"error": f"File not found: {file_path}"}
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}

        # Cache all lines for agent callbacks (fetch_lines_around). With a cap,
        # this is the recent window — search/context-fetch are scoped to it.
        self._last_file = file_path
        self._all_lines = lines

        # No date filter -> group all lines into entries and return
        no_filter = not (self.relative_day or (self.from_dt and self.to_dt) or self.date)
        if no_filter:
            entries = self._group_into_entries(lines)
            result = {
                "file": file_path,
                "filter": None,
                "total_matched": len(entries),
                "logs": entries,
            }
            if self._capped:
                result["truncated"] = True
            return result

        pick_date = self._pick_datetime()

        # Timestamp extraction is centralized in log_format (single source).
        extract_timestamp = log_format.extract_timestamp

        def line_matches(line: str, check_fn) -> Optional[bool]:
            ts = extract_timestamp(line)
            if ts is None:
                return None  # no timestamp → continuation line
            return check_fn(ts)

        if pick_date is None:
            return {"error": "No date provided"}
        if isinstance(pick_date, str) and pick_date.startswith("Invalid"):
            return {"error": pick_date}

        # Build check function
        if isinstance(pick_date, str) and len(pick_date) == 10:
            target_date = datetime.strptime(pick_date, "%Y-%m-%d").date()

            def check(ts):
                return ts.date() == target_date

        elif isinstance(pick_date, str):
            target_dt = datetime.strptime(pick_date, "%Y-%m-%d %H:%M:%S")

            def check(ts):
                return abs((ts - target_dt).total_seconds()) <= 60

        elif isinstance(pick_date, dict):
            from_dt = datetime.strptime(pick_date["from"], "%Y-%m-%d %H:%M:%S")
            to_dt = datetime.strptime(pick_date["to"], "%Y-%m-%d %H:%M:%S")

            def check(ts):
                return from_dt <= ts <= to_dt

        else:
            return {"error": f"Unsupported pick_date type: {type(pick_date)}"}

        # Filter lines — keep continuation lines (no timestamp) after a matched line
        matched_lines = []
        inside_match = False
        for line in lines:
            result = line_matches(line, check)
            if result is True:
                inside_match = True
                matched_lines.append(line.rstrip())
            elif result is None and inside_match:
                matched_lines.append(line.rstrip())  # continuation/traceback line
            elif result is False:
                inside_match = False  # new timestamped line, didn't match

        # Group into logical entries (timestamp + continuation lines)
        entries = self._group_into_entries(matched_lines)

        result = {
            "file": file_path,
            "filter": pick_date,
            "total_matched": len(entries),
            "logs": entries,
        }
        if self._capped:
            result["truncated"] = True
        return result

    def fetch_lines_around(self, timestamp: str, context_lines: int = 10) -> dict:
        """
        Agent-to-agent callback.
        Called by ContextBridge when Qwen says it needs more context
        around a specific timestamp.

        Args:
            timestamp:     "2024-03-01 10:00:05" — the point Qwen wants context around
            context_lines: How many lines before and after to return (default 10)

        Returns:
            {
                "timestamp":     str,
                "context_lines": int,
                "lines":         str   ← extra context to feed back to Qwen
                "found":         bool
            }
        """
        if not self._all_lines:
            return {"error": "No file loaded yet. Call fetch_logs_by_date() first."}

        extract_ts = log_format.extract_timestamp

        # Parse the requested timestamp
        target = None
        for _pattern, fmt in log_format.TS_PATTERNS:
            try:
                target = datetime.strptime(timestamp.strip(), fmt)
                break
            except ValueError:
                continue

        if target is None:
            return {"error": f"Could not parse timestamp: {timestamp}"}

        # Find the closest line to that timestamp
        best_idx = None
        best_diff = float("inf")
        for i, line in enumerate(self._all_lines):
            ts = extract_ts(line)
            if ts:
                diff = abs((ts - target).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

        if best_idx is None or best_diff > 300:  # >5 min away = not found
            return {
                "timestamp": timestamp,
                "context_lines": context_lines,
                "lines": "",
                "found": False,
                "truncated": self._capped,
            }

        # Grab surrounding lines
        start = max(0, best_idx - context_lines)
        end = min(len(self._all_lines), best_idx + context_lines + 1)
        surrounding = self._all_lines[start:end]

        return {
            "timestamp": timestamp,
            "context_lines": context_lines,
            "lines": "\n".join(line_item.rstrip() for line_item in surrounding),
            "found": True,
            "truncated": self._capped,
        }

    def search_logs(self, keyword: str, max_results: int = 20) -> dict:
        """
        Search all loaded log lines for a keyword or identifier.
        Returns matches in REVERSE order (most recent first).

        Args:
            keyword:     Any string — INC1033234, "connection refused", "db:5432"
            max_results: Max entries to return. Default 20.

        Returns:
            {
                "keyword":     str,
                "total_found": int,
                "entries":     List[str]  ← matched grouped entries, recent first
            }
        """
        logger.debug("search_logs: keyword=%s, max_results=%d", keyword, max_results)

        if not self._all_lines:
            return {"error": "No file loaded yet. Call fetch_logs_by_date() first."}

        keyword_lower = keyword.lower()

        # Group all lines into entries first
        all_entries = self._group_into_entries(self._all_lines)

        # Filter entries containing the keyword
        matched = [entry for entry in all_entries if keyword_lower in entry.lower()]

        # Reverse — most recent first
        matched = list(reversed(matched))[:max_results]
        logger.debug("search_logs: %d entries scanned, %d matched", len(all_entries), len(matched))

        result = {
            "keyword": keyword,
            "total_found": len(matched),
            "entries": matched,
        }
        if self._capped:
            result["truncated"] = True
        return result

    def _group_into_entries(self, lines: list) -> list:
        """
        Group raw lines into logical entries.
        Each entry = one timestamped line + all following continuation
        lines (tracebacks, multiline errors) as a single string.

        Example:
            Input lines:
                "2024-03-01 10:00:07 ERROR Reconnect failed"
                "Traceback (most recent call last):"
                "  File app.py, line 42"
                "ConnectionError: timed out"
                "2024-03-01 10:00:09 INFO  Retrying..."

            Output entries:
                [
                  "2024-03-01 10:00:07 ERROR Reconnect failed\nTraceback...\n  File app.py...\nConnectionError: timed out",
                  "2024-03-01 10:00:09 INFO  Retrying..."
                ]
        """
        has_timestamp = log_format.has_timestamp

        entries = []
        current: list[str] = []

        for line in lines:
            if has_timestamp(line):
                if current:
                    entries.append("\n".join(current))
                current = [line.rstrip()]
            else:
                # Continuation line (traceback etc.) — attach to current entry
                if current:
                    current.append(line.rstrip())
                # If no current entry yet, skip (header lines before first timestamp)

        if current:
            entries.append("\n".join(current))

        return entries
