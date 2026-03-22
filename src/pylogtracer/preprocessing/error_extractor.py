"""
error_extractor.py
===================
Takes a LIST of grouped log entries from smart_reader and:
  1. Filters entries that contain errors
  2. Parses each into structured format (primary_error, traceback, timestamp)
  3. Classifies error_type via ErrorTypeClassifier (regex first, LLM fallback)
  4. Clusters errors by time proximity — duplicate types across batches
     are automatically MERGED into the same cluster
  5. Counts error frequency across all entries
  6. Returns the LAST cluster as the most recent incident

Usage:
    extractor = ErrorExtractor(gap_seconds=60, classifier=classifier)
    result    = extractor.extract(log_entries)   # log_entries = smart_reader result["logs"]
    last      = result["last_cluster"]           # → send to RootCauseAnalyzer
    frequency = result["frequency"]              # → error frequency table
"""

import re
from typing import List, Dict, Optional
from datetime import datetime
from collections import Counter

from pylogtracer.preprocessing.error_type_classifier import ErrorTypeClassifier


class ErrorExtractor:
    """
    Extract structured error chains from a list of grouped log entries.
    Each entry is a string (one timestamp line + its continuation lines).
    """

    ERROR_WORDS = ["error", "exception", "failed", "critical", "fatal"]
    TRACEBACK_WORD = "traceback"

    TIMESTAMP_PATTERNS = [
        (r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "%Y-%m-%d %H:%M:%S"),
        (r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}", "%d-%m-%Y %H:%M:%S"),
        (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "%Y-%m-%dT%H:%M:%S"),
        (r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}", "%Y/%m/%d %H:%M:%S"),
    ]

    def __init__(
        self,
        gap_seconds: int = 60,
        classifier: Optional[ErrorTypeClassifier] = None,
    ):
        """
        Args:
            gap_seconds:  Time gap (seconds) to separate incidents. Default 60s.
            classifier:   ErrorTypeClassifier instance. If None, uses regex only.
        """
        self.gap_seconds = gap_seconds
        self.classifier = classifier  # None = regex-only mode

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def extract(self, log_entries: List[str]) -> Dict:
        """
        Full extraction pipeline.

        Args:
            log_entries: List of grouped entry strings from smart_reader["logs"].
                         Each item = one timestamped record + continuation lines.

        Returns:
            {
                "all_errors"    : list of every parsed error entry,
                "clusters"      : errors grouped into time-based incidents,
                "last_cluster"  : most recent incident → send to RootCauseAnalyzer,
                "frequency"     : { error_type: count } sorted by most frequent,
                "total_errors"  : int
            }
        """
        # Step 1 — Filter and parse error entries
        all_errors = []
        for entry in log_entries:
            if self._entry_has_error(entry):
                parsed = self._parse_error_entry(entry)
                all_errors.append(parsed)

        if not all_errors:
            return {
                "all_errors": [],
                "clusters": [],
                "last_cluster": None,
                "frequency": {},
                "total_errors": 0,
            }

        # Step 2 — Classify error types
        # (regex first → LLM fallback for unknowns → duplicates flagged)
        if self.classifier:
            all_errors = self.classifier.classify(all_errors)
        # else: error_type stays as regex result set in _parse_error_entry

        # Step 3 — Cluster by time + merge duplicate types across batches
        clusters = self._cluster_and_merge(all_errors)

        # Step 4 — Frequency count
        frequency = self._count_frequency(all_errors)

        return {
            "all_errors": all_errors,
            "clusters": clusters,
            "last_cluster": clusters[-1] if clusters else None,
            "frequency": frequency,
            "total_errors": len(all_errors),
        }

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — error detection
    # ─────────────────────────────────────────────────────────────

    def _entry_has_error(self, entry: str) -> bool:
        return any(w in entry.lower() for w in self.ERROR_WORDS)

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — parsing
    # ─────────────────────────────────────────────────────────────

    def _parse_error_entry(self, entry: str) -> Dict:
        """
        Parse a single grouped entry into structured format.
        error_type is set by regex here — classifier may override it later.
        """
        chain = self._extract_chain(entry)
        timestamp = self._extract_timestamp(entry)
        # Regex classification as initial value — classifier overrides if needed
        error_type = self._regex_classify(chain["primary"])

        return {
            "primary_error": chain["primary"],
            "traceback": chain["traceback"],
            "timestamp": timestamp,
            "error_type": error_type,
            "type_source": "regex" if error_type != "UnknownError" else "pending",
            "is_duplicate": False,
            "full_entry": entry.strip(),
        }

    def _extract_chain(self, entry: str) -> Dict:
        """
        Scan all lines of an entry:
          primary_error = first line with an error word
          traceback     = everything from 'Traceback' onwards
        Traceback is guaranteed to be here because smart_reader grouped it.
        """
        lines = entry.splitlines()
        primary_error = None
        traceback_lines = []
        traceback_started = False

        for line in lines:
            lower = line.lower()

            if not primary_error and any(w in lower for w in self.ERROR_WORDS):
                primary_error = line.strip()

            if self.TRACEBACK_WORD in lower:
                traceback_started = True
            if traceback_started:
                traceback_lines.append(line)

        return {
            "primary": primary_error or "unknown_error",
            "traceback": "\n".join(traceback_lines),
        }

    def _regex_classify(self, error_line: str) -> str:
        """
        Fast regex classification for named exceptions.
        Returns "UnknownError" if no named exception found.
        """
        match = re.search(r"\b([A-Z][a-zA-Z]+(?:Error|Exception|Warning|Critical|Fatal))\b", error_line)
        return match.group(1) if match else "UnknownError"

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — timestamp
    # ─────────────────────────────────────────────────────────────

    def _extract_timestamp(self, entry: str) -> Optional[datetime]:
        for pattern, fmt in self.TIMESTAMP_PATTERNS:
            m = re.search(pattern, entry)
            if m:
                try:
                    return datetime.strptime(m.group(), fmt)
                except ValueError:
                    continue
        return None

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — clustering with duplicate merge
    # ─────────────────────────────────────────────────────────────

    def _cluster_and_merge(self, errors: List[Dict]) -> List[List[Dict]]:  # noqa: C901
        """
        Group errors into clusters by TWO rules:

        Rule 1 — Time proximity:
            Two consecutive errors within gap_seconds → same cluster.

        Rule 2 — Duplicate type merge:
            If an error's type was already seen in a DIFFERENT (earlier) cluster
            → pull it INTO that earlier cluster regardless of time gap.
            This handles the case where the same root error recurs across batches.

        Result: errors of the same type always end up in the same cluster,
        even if they were far apart in time.
        """
        if not errors:
            return []

        # First pass — time-based clustering
        clusters = []
        current = [errors[0]]

        for prev, curr in zip(errors, errors[1:]):
            tp = prev["timestamp"]
            tc = curr["timestamp"]

            if tp and tc:
                split = abs((tc - tp).total_seconds()) > self.gap_seconds
            else:
                split = False  # no timestamps → keep together

            if split:
                clusters.append(current)
                current = [curr]
            else:
                current.append(curr)
        clusters.append(current)

        # Second pass — merge clusters that share the same error_type
        # Build: error_type → cluster index of first occurrence
        type_to_cluster: Dict[str, int] = {}
        merged_clusters: List[Optional[List]] = list(clusters)

        for ci, cluster in enumerate(merged_clusters):
            if cluster is None:
                continue
            for error in cluster:
                etype = error["error_type"]
                if etype == "UnknownError":
                    continue  # don't merge unknowns blindly

                if etype not in type_to_cluster:
                    # First time seeing this type — register this cluster
                    type_to_cluster[etype] = ci
                else:
                    # Already seen in an earlier cluster → merge into it
                    target_ci = type_to_cluster[etype]
                    if target_ci != ci and merged_clusters[target_ci] is not None:
                        print(f"  [Extractor] Merging cluster {ci} into cluster " f"{target_ci} (duplicate type: {etype})")
                        target_cluster = merged_clusters[target_ci]
                        if target_cluster is not None:
                            target_cluster.extend(cluster)
                        merged_clusters[ci] = None  # mark as absorbed
                        break

        # Clean up absorbed clusters + sort each cluster by timestamp
        final = []
        for cluster in merged_clusters:
            if cluster is None:
                continue
            # Sort by timestamp within each cluster
            cluster.sort(key=lambda e: e["timestamp"] or datetime.min)
            final.append(cluster)

        return final

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — frequency
    # ─────────────────────────────────────────────────────────────

    def _count_frequency(self, errors: List[Dict]) -> Dict:
        """
        Count occurrences of each error_type.
        Returns dict sorted by most frequent first.
        """
        counts = Counter(e["error_type"] for e in errors)
        return dict(counts.most_common())
