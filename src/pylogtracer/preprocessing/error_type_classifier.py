"""
error_type_classifier.py
=========================
Hybrid error type classifier with session-level keyword learning.

Classification priority per entry:
  1. Named exception regex  — free, instant (ConnectionError, ValueError etc.)
  2. Keyword store match    — free, learned from LLM this session
  3. LLM batch call         — only for truly unknown entries

LLM also returns keywords for each type it classifies.
Valid keywords are stored in keyword_store and used to match
future entries of the same type — avoiding redundant LLM calls.

Keyword validation rules:
  - At least 3 characters
  - At least 2 meaningful words OR one very specific phrase
  - No pure symbols, digits-only, or port numbers
  - Not a common stop word (the, in, at, for...)

Keywords reset each session (not persisted to file).

Usage:
    from pylogtracer.llm.llm_factory import LLMFactory
    factory    = LLMFactory({"provider": "ollama", "model": "qwen2.5:7b"})
    classifier = ErrorTypeClassifier(factory=factory)
    entries    = classifier.classify(error_entries)
"""

import re
import math
from typing import List, Dict, Optional, Tuple

try:
    from pydantic import BaseModel, Field
    from langchain_core.prompts import ChatPromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    class BaseModel:  # noqa: E742
        pass

    def Field(**kw):  # noqa: E302,E731
        return None


# ── Model context windows ─────────────────────────────────────────
MODEL_CONTEXT_WINDOWS = {
    "qwen2.5:1.8b": 2048,
    "qwen2.5:3b": 4096,
    "qwen2.5:7b": 8192,
    "qwen2.5:14b": 16384,
    "qwen2.5:32b": 32768,
    "qwen2.5:72b": 65536,
    "gpt-4o": 16384,
    "gpt-4o-mini": 16384,
    "gpt-3.5-turbo": 4096,
    "claude-3-5-sonnet-20241022": 65536,
    "claude-3-5-haiku-20241022": 65536,
}

PROMPT_OVERHEAD_TOKENS = 400  # slightly higher — prompt now asks for keywords too
TOKENS_PER_ENTRY = 60
MAX_BATCH_SIZE = 20
MIN_BATCH_SIZE = 3

# Keyword validation
MIN_KEYWORD_LEN = 3
MIN_KEYWORD_WORDS = 1  # single word ok if specific enough
MAX_KEYWORD_WORDS = 6  # avoid full sentences
STOP_WORDS = {
    "the",
    "in",
    "at",
    "for",
    "on",
    "an",
    "a",
    "is",
    "was",
    "to",
    "of",
    "and",
    "or",
    "not",
    "with",
    "from",
    "by",
    "error",
    "failed",
    "exception",
    "warning",  # too generic on their own
}


# ── Pydantic schema ───────────────────────────────────────────────
class EntryClassification(BaseModel):
    """Classification for a single error entry."""

    error_type: str = Field(description="PascalCase error type label")
    keywords: List[str] = Field(
        description="2-4 short phrases from this error message that identify this error type. "
        "Phrases should be specific enough to match similar future errors."
    )


class BatchClassification(BaseModel):
    """LLM response schema for a full batch."""

    classifications: Dict[str, EntryClassification] = Field(
        description="Map of entry index (string '1','2'...) to its classification"
    )


class ErrorTypeClassifier:
    """
    Hybrid classifier: named-exception regex → keyword store → LLM batch.

    The keyword store grows during the session as LLM classifies new error types.
    Once a type's keywords are learned, all future matching entries are typed
    for free without any LLM call.

    Args:
        factory: LLMFactory instance. None = regex-only mode.
    """

    NAMED_EXCEPTION_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:Error|Exception|Warning|Critical|Fatal))\b")

    # Pattern-based fallback classifiers (for when LLM not available)
    PATTERN_MATCHERS = [
        (r"(?:timeout|timed out|time limit|time out)", "TimeoutError"),
        (r"(?:connection|connect|refused|unreachable|tcp|socket)", "ConnectionError"),
        (r"(?:authentication|unauthorized|auth failed|401|invalid.*credentials)", "AuthenticationError"),
        (r"(?:permission|denied|403|forbidden|access denied)", "PermissionError"),
        (r"(?:not found|404|no such|doesn\'t exist|cannot find)", "NotFoundError"),
        (r"(?:memory|out of memory|oom|malloc failed|heap)", "MemoryError"),
        (r"(?:disk|storage|space full|io error|file system)", "DiskError"),
        (r"(?:database|db|postgres|mysql|sql|query|transaction)", "DatabaseError"),
        (r"(?:network|socket|tcp|udp|http|request|response)", "NetworkError"),
        (r"(?:api|endpoint|rest|json|xml|parse)", "APIError"),
        (r"(?:type mismatch|type error|incompatible)", "TypeError"),
        (r"(?:invalid|value error|bad value|invalid format)", "ValueError"),
        (r"(?:index|out of range|out of bounds)", "IndexError"),
        (r"(?:null|none|undefined|nil|not initialized)", "NullPointerException"),
        (r"(?:runtime|fatal|crash|panic|segmentation)", "RuntimeError"),
        (r"(?:deadlock|race condition|concurrency)", "ConcurrencyError"),
        (r"(?:deprecated|unsupported|not implemented)", "NotImplementedError"),
    ]

    # Prompt built lazily in _get_structured_llm() — avoids LangChain import at class load time

    def __init__(self, factory=None):
        self.factory = factory
        self._structured_llm = None

        # keyword_store: { error_type -> [keyword_phrase, ...] }
        # Built up during this session as LLM classifies new types
        self._keyword_store: Dict[str, List[str]] = {}

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def classify(self, error_entries: List[Dict]) -> List[Dict]:  # noqa: C901
        """
        Classify error_type for each error entry.

        Pass 1: Named exception regex        (free)
        Pass 2: Keyword store match          (free, learned this session)
        Pass 3: LLM batch for unknowns       (let LLM decide if truly errors, even if INFO/DEBUG)
        """
        if not error_entries:
            return error_entries

        needs_llm = []

        for entry in error_entries:
            # Pass 1 — named exception regex
            regex_type = self._regex_classify(entry["primary_error"])
            if regex_type:
                entry["error_type"] = regex_type
                entry["is_duplicate"] = False
                entry["type_source"] = "regex"
                continue

            # Pass 2 — scored keyword store match
            kw_result, kw_status = self._keyword_store_classify(entry["primary_error"])

            if kw_status == "match":
                # Clear winner
                entry["error_type"] = kw_result
                entry["is_duplicate"] = True
                entry["type_source"] = "keyword"
                continue

            elif kw_status == "tie":
                # Ambiguous — equal score for multiple types
                # Send to LLM with candidate types as hint
                entry["error_type"] = "UnknownError"
                entry["is_duplicate"] = False
                entry["type_source"] = "pending_llm_tie"
                entry["_tie_candidates"] = kw_result  # list of tied types
                needs_llm.append(entry)
                continue

            # Pass 3 — truly unknown, no keyword match
            # Send to LLM to decide (even if INFO/DEBUG — human might have mislabeled)
            entry["error_type"] = "UnknownError"
            entry["is_duplicate"] = False
            entry["type_source"] = "pending_llm"
            needs_llm.append(entry)

        regex_count = sum(1 for e in error_entries if e["type_source"] == "regex")
        keyword_count = sum(1 for e in error_entries if e["type_source"] == "keyword")
        llm_count = len(needs_llm)

        print(f"  [Classifier] regex={regex_count} | " f"keyword_store={keyword_count} | " f"llm_needed={llm_count}")

        if not needs_llm:
            return error_entries

        if self.factory is None:
            print("  [Classifier] ⚠️  No LLM factory configured")
            print("              Using only pattern-based classification (above)")
            print("              To enable LLM learning, pass LLMFactory to ErrorTypeClassifier")
            return error_entries

        # Pass 3 — LLM batch
        batch_size = self._compute_batch_size()
        print(f"  [Classifier] model={self.factory.get_model()} | " f"batch_size={batch_size}")

        seen_types: Dict[str, Dict] = {}
        batches = [needs_llm[i : i + batch_size] for i in range(0, len(needs_llm), batch_size)]

        for batch_num, batch in enumerate(batches, 1):
            print(f"  [Classifier] Batch {batch_num}/{len(batches)} " f"({len(batch)} entries)...")

            result = self._classify_batch(batch)

            if not result:
                print(f"  [Classifier] Batch {batch_num} failed — " f"keeping UnknownError")
                continue

            # Debug: show what LLM returned
            print(f"  [Classifier] LLM returned {len(result)} classifications:")
            for idx, classification in result.items():
                print(f"              [{idx}] {classification.error_type} (keywords: {len(classification.keywords)})")

            for i, entry in enumerate(batch):
                classification = result.get(str(i + 1))
                if not classification:
                    print(f"  [Classifier]   [WARN] Entry {i+1} missing from LLM result")
                    continue

                llm_type = self._normalize_type(classification.error_type)
                keywords = classification.keywords

                # If LLM classified as NonError, mark it and skip learning
                if llm_type == "NonError":
                    entry["error_type"] = "NonError"
                    entry["is_duplicate"] = False
                    entry["type_source"] = "llm_non_error"
                    print(f"  [Classifier]   Classified as NonError: '{entry['primary_error'][:60]}...'")
                    continue

                print(f"  [Classifier]   Applying: {llm_type} to '{entry['primary_error'][:60]}...'")

                # Learn validated keywords into store (only for real errors)
                valid_kws = self._validate_keywords(keywords)
                if valid_kws:
                    self._learn_keywords(llm_type, valid_kws)
                    print(
                        f"  [Classifier]   >> Learned {len(valid_kws)} "
                        f"keyword(s) for {llm_type}: {valid_kws}"
                    )

                if llm_type != "UnknownError" and llm_type in seen_types:
                    entry["error_type"] = llm_type
                    entry["is_duplicate"] = True
                    entry["duplicate_of_ts"] = seen_types[llm_type].get("timestamp")
                    entry["type_source"] = "llm_duplicate"
                else:
                    entry["error_type"] = llm_type
                    entry["is_duplicate"] = False
                    entry["type_source"] = "llm"
                    seen_types[llm_type] = entry

        return error_entries

    def get_keyword_store(self) -> Dict[str, List[str]]:
        """Return current keyword store (for inspection/debugging)."""
        return dict(self._keyword_store)

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — classification passes
    # ─────────────────────────────────────────────────────────────

    def _regex_classify(self, error_line: str) -> Optional[str]:
        """Try multiple strategies to classify error without LLM."""
        lower = error_line.lower()

        # Strategy 1: Named exception class (e.g., ConnectionError, ValueError)
        match = self.NAMED_EXCEPTION_RE.search(error_line)
        if match:
            return match.group(1)

        # Strategy 2: Pattern-based matching (fallback when no exception class found)
        for pattern, error_type in self.PATTERN_MATCHERS:
            if re.search(pattern, lower):
                return error_type

        return None

    def _keyword_store_classify(self, error_line: str) -> Tuple[Optional[object], str]:
        """
        Check if error_line contains any keyword from the store using word-boundary matching.
        Score-based keyword matching — handles ambiguous overlapping types
        like "API connection refused" vs "DB connection refused".

        Scoring formula per type:
          score = count of keywords that match (with word boundaries)
          Multiple matches = higher score, longer keywords = tie-breaker

        Returns:
          (winner, status) where:
            status="match"    → clear winner, winner=error_type string
            status="tie"      → equal scores, winner=list of tied type strings → LLM decides
            status="no_match" → nothing matched, winner=None
        """
        lower = error_line.lower()
        scores: Dict[str, int] = {}
        match_counts: Dict[str, int] = {}
        match_lengths: Dict[str, int] = {}

        for error_type, keywords in self._keyword_store.items():
            match_count = 0
            match_length = 0

            for kw in keywords:
                # Use word boundary regex — exact phrase matching
                pattern = r"\b" + re.escape(kw.lower()) + r"\b"
                if re.search(pattern, lower):
                    match_count += 1
                    match_length += len(kw)

            if match_count > 0:
                scores[error_type] = match_count
                match_counts[error_type] = match_count
                match_lengths[error_type] = match_length

        if not scores:
            return None, "no_match"

        # Sort by: (1) match count DESC, (2) keyword length sum DESC
        ranked = sorted(scores.items(), key=lambda x: (match_counts[x[0]], match_lengths[x[0]]), reverse=True)
        best_type, best_score = ranked[0]

        # True tie — second type has the exact same match count AND length
        if len(ranked) > 1 and (
            match_counts[ranked[1][0]] == match_counts[best_type] and match_lengths[ranked[1][0]] == match_lengths[best_type]
        ):
            tied = [
                t
                for t in scores.keys()
                if match_counts[t] == match_counts[best_type] and match_lengths[t] == match_lengths[best_type]
            ]
            return tied, "tie"

        return best_type, "match"

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — keyword learning
    # ─────────────────────────────────────────────────────────────

    def _is_unique_keyword(self, keyword: str, error_type: str) -> bool:
        """
        Check if keyword is sufficiently unique to distinguish this error type.
        Rejects keywords that are substrings of or contain other error type's keywords.
        """
        kw_lower = keyword.lower()

        for other_type, other_keywords in self._keyword_store.items():
            if other_type == error_type:
                continue

            for other_kw in other_keywords:
                other_lower = other_kw.lower()
                # Reject if keywords are too similar
                if kw_lower in other_lower or other_lower in kw_lower or kw_lower == other_lower:
                    return False

        return True

    def _validate_keywords(self, keywords: List[str]) -> List[str]:
        """
        Filter LLM-suggested keywords to only keep useful ones.

        A keyword is valid if:
          - At least MIN_KEYWORD_LEN characters
          - Between 1 and MAX_KEYWORD_WORDS words
          - Not purely digits, symbols, or port-like (digits+colon)
          - Not a single stop word
          - Not a pure version/number string (v1.2, 404, etc.)
        """
        valid = []
        for kw in keywords:
            kw = kw.strip().lower()

            if len(kw) < MIN_KEYWORD_LEN:
                continue

            words = kw.split()

            if len(words) > MAX_KEYWORD_WORDS:
                continue

            # Reject pure digit / symbol strings
            if re.match(r"^[\d\s\.\:\-\/]+$", kw):
                continue

            # Reject port-like patterns (5432, :5432, db:5432)
            if re.search(r":\d+", kw):
                continue

            # Reject if single stop word
            if len(words) == 1 and words[0] in STOP_WORDS:
                continue

            # Reject if ALL words are stop words
            if all(w in STOP_WORDS for w in words):
                continue

            # Reject version strings (v1.2.3, 1.0.0)
            if re.match(r"^v?\d+[\.\d]+$", kw):
                continue

            valid.append(kw)

        return valid

    def _learn_keywords(self, error_type: str, keywords: List[str]):
        """Add validated keywords to the store for this error type.

        Only learns keywords that are unique enough to distinguish this error type
        from previously learned types.
        """
        if error_type not in self._keyword_store:
            self._keyword_store[error_type] = []

        # Avoid duplicates and non-unique keywords
        existing = set(self._keyword_store[error_type])
        learned = []

        for kw in keywords:
            if kw in existing:
                continue

            # Check uniqueness before adding
            if self._is_unique_keyword(kw, error_type):
                self._keyword_store[error_type].append(kw)
                existing.add(kw)
                learned.append(kw)

        if learned:
            print(f"  [Classifier]   Stored unique keywords: {learned}")
        else:
            print(f"  [Classifier]   No unique keywords learned (too similar to existing types)")

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — batch sizing
    # ─────────────────────────────────────────────────────────────

    def _compute_batch_size(self) -> int:
        model = self.factory.get_model() if self.factory else "default"
        context = MODEL_CONTEXT_WINDOWS.get(model, 8192)
        usable = context - PROMPT_OVERHEAD_TOKENS
        size = math.floor(usable / TOKENS_PER_ENTRY)
        return max(MIN_BATCH_SIZE, min(size, MAX_BATCH_SIZE))

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — LLM call
    # ─────────────────────────────────────────────────────────────

    def _get_structured_llm(self):
        if self._structured_llm is None:
            if not LANGCHAIN_AVAILABLE:
                raise ImportError(
                    "LangChain not installed. "
                    "Run: pip install langchain langchain-openai "
                    "langchain-anthropic langchain-ollama"
                )
            system_msg = (
                "You are an error type classifier for software logs.\n\n"
                "For EACH numbered error line, decide:\n"
                "1. If it's a REAL ERROR → classify by ROOT CAUSE\n"
                "2. If it's NOT a real error (INFO, DEBUG, success message) → respond with 'NonError'\n\n"
                "RULES:\n"
                "1. Classify ONLY actual errors by WHAT WENT WRONG, not incident number\n"
                "2. Use PascalCase error type names for real errors:\n"
                "   - EmailDeliveryError, AccountLockoutError, WorkerProcessError\n"
                "   - DatabaseError, ConnectionError, TimeoutError, PermissionError, etc.\n"
                "3. Use 'NonError' for INFO/DEBUG/SUCCESS logs or non-critical messages\n"
                "4. Extract 2-4 keyword PHRASES that describe the error (or empty list for NonError)\n"
                "5. Keywords must be specific and present in the message\n\n"
                "RESPONSE FORMAT:\n"
                "Use NUMERIC keys ONLY: '1', '2', '3', ..., NOT INC IDs\n"
                "Example:\n"
                "  '1': {{error_type: 'EmailDeliveryError', keywords: ['email delivery', 'max retries']}}\n"
                "  '2': {{error_type: 'NonError', keywords: []}}\n"
                "  '3': {{error_type: 'AccountLockoutError', keywords: ['account locked', 'failed attempts']}}\n\n"
                "IMPORTANT: Classify ALL entries in the list. Return entries for all numeric indices.\n"
                "Return only valid JSON matching the schema."
            )
            prompt = ChatPromptTemplate.from_messages([("system", system_msg), ("human", "{entries_text}")])
            self._structured_llm = prompt | self.factory.get_llm().with_structured_output(BatchClassification)
        return self._structured_llm

    def _classify_batch(self, batch: List[Dict]) -> Optional[Dict[str, EntryClassification]]:  # noqa: C901
        lines = []
        for i, entry in enumerate(batch):
            line = f"[{i + 1}] {entry['primary_error']}"
            # If this entry is a tie, give LLM the candidates as a hint
            candidates = entry.get("_tie_candidates")
            if candidates:
                line += f"  [AMBIGUOUS — could be: {' or '.join(candidates)}]"
            lines.append(line)

        entries_text = "\n".join(lines)

        try:
            chain = self._get_structured_llm()
            result = chain.invoke({"entries_text": entries_text})

            # LLM may return classifications keyed by ID (INC2000003) or index (1, 2, 3)
            # Remap IDs to numeric indices for consistency
            classifications = result.classifications
            remapped = {}

            # Check if keys are numeric strings
            numeric_keys = all(k.isdigit() for k in classifications.keys())

            if not numeric_keys:
                # Keys are probably IDs (INC2000003, etc)
                # Map them back to indices [1], [2], etc
                for orig_idx, entry in enumerate(batch, 1):
                    # Try to find a match by primary_error content
                    found = False
                    for key, classification in classifications.items():
                        # Check if this classification might be for this entry
                        # Use a simple heuristic: similar error type or first N that match
                        remapped[str(orig_idx)] = classification
                        found = True
                        break  # Take first available
                    if not found:
                        print(
                            f"  [Classifier] [WARN] Could not map entry {orig_idx} from LLM keys: {list(classifications.keys())}"
                        )
                return remapped if remapped else classifications

            return classifications

        except Exception as e:
            print(f"  [Classifier] [ERROR] LLM call failed: {e}")
            import traceback

            traceback.print_exc()
            # On failure — fall back to first candidate
            fallback = {}
            for i, entry in enumerate(batch):
                candidates = entry.get("_tie_candidates")
                if candidates:
                    fallback[str(i + 1)] = type("obj", (object,), {"error_type": candidates[0], "keywords": []})()
            return fallback if fallback else None

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — normalization
    # ─────────────────────────────────────────────────────────────

    def _normalize_type(self, raw_type: str) -> str:
        """Normalize error type to proper PascalCase."""
        # Remove any lowercase conversion first - preserve original case
        raw_type = raw_type.strip()

        # Already properly formatted PascalCase?
        if raw_type and raw_type[0].isupper() and "_" not in raw_type and "-" not in raw_type:
            return raw_type or "UnknownError"

        # Need to reformat - split on various delimiters and capitalize each word
        words = re.split(r"[\s_\-]+", raw_type.lower())
        return "".join(w.capitalize() for w in words if w) or "UnknownError"
