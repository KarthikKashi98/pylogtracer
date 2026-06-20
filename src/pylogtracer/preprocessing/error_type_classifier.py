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
import os
import json
import math
import logging
import tempfile
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from pydantic import BaseModel, Field  # type: ignore
    from langchain_core.prompts import ChatPromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    class BaseModel:  # noqa: E742, type: ignore
        pass

    def Field(**kw):  # noqa: E302, type: ignore  # pragma: no cover
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

    # Volatile tokens stripped when dedup_strip_ids=True, so that lines
    # differing only by incident/request id or UUID collapse to one LLM call.
    _VOLATILE_RE = re.compile(
        r"\b(INC\d+|REQ-?\w+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        factory=None,
        cache_path: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        max_entry_chars: int = 500,
        dedup: bool = True,
        dedup_strip_ids: bool = False,
        redactor=None,
    ):
        """
        Args:
            factory:            LLMFactory instance. None = regex/pattern-only.
            cache_path:         JSON file to persist the learned keyword store
                                across runs. None = in-memory only (default).
            max_context_tokens: Override the model context window used to size
                                LLM batches. None = use MODEL_CONTEXT_WINDOWS.
            max_entry_chars:    Per-line cap on the text sent to the LLM (the
                                stored entry is never mutated). Default 500.
            dedup:              Classify identical unknown lines once and fan the
                                result out to duplicates. Default True.
            dedup_strip_ids:    When deduping, also collapse lines that differ
                                only by incident/request id or UUID. Default False.
        """
        self.factory = factory
        self._structured_llm = None
        self.cache_path = cache_path
        self.max_context_tokens = max_context_tokens
        self.max_entry_chars = max_entry_chars
        self.dedup = dedup
        self.dedup_strip_ids = dedup_strip_ids
        self.redactor = redactor  # optional callable(text)->text for the LLM boundary
        self._dirty = False  # set when new keywords are learned; gates save()

        # keyword_store: { error_type -> [keyword_phrase, ...] }
        # Built up during this session as LLM classifies new types; optionally
        # loaded from / saved to cache_path so it survives across runs.
        self._keyword_store: Dict[str, List[str]] = {}
        if cache_path:
            self._load_keyword_store()

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

        logger.info("[Classifier] regex=%d | keyword_store=%d | llm_needed=%d", regex_count, keyword_count, llm_count)

        if not needs_llm:
            return error_entries

        if self.factory is None:
            logger.info(
                "[Classifier] No LLM configured — using pattern-based classification only. "
                "Pass an llm_config to LogTracer to enable LLM classification and keyword learning."
            )
            return error_entries

        # Within-run dedup: classify each distinct primary_error ONCE and fan
        # the result out to its duplicates. Tie entries carry per-entry hints,
        # so they are always classified individually (never deduped).
        dup_groups: Dict[str, List[Dict]] = {}
        if self.dedup:
            representatives = []
            for entry in needs_llm:
                if entry.get("_tie_candidates"):
                    representatives.append(entry)
                    continue
                key = self._dedup_key(entry["primary_error"])
                if key in dup_groups:
                    dup_groups[key].append(entry)
                else:
                    dup_groups[key] = [entry]
                    representatives.append(entry)
            to_classify = representatives
        else:
            to_classify = needs_llm

        if self.dedup and len(to_classify) < len(needs_llm):
            logger.info(
                "[Classifier] dedup: %d unknown lines -> %d unique sent to LLM",
                len(needs_llm), len(to_classify),
            )

        # Pass 3 — LLM batch
        batch_size = self._compute_batch_size()
        logger.info("[Classifier] model=%s | batch_size=%d", self.factory.get_model(), batch_size)

        seen_types: Dict[str, Dict] = {}
        batches = [to_classify[i : i + batch_size] for i in range(0, len(to_classify), batch_size)]

        for batch_num, batch in enumerate(batches, 1):
            logger.info("[Classifier] Batch %d/%d (%d entries)...", batch_num, len(batches), len(batch))

            result = self._classify_batch(batch)

            if not result:
                logger.warning("[Classifier] Batch %d failed — keeping UnknownError", batch_num)
                continue

            logger.debug("[Classifier] LLM returned %d classifications", len(result))
            for idx, classification in result.items():
                logger.debug("              [%s] %s (keywords: %d)", idx, classification.error_type, len(classification.keywords))

            for i, entry in enumerate(batch):
                classification_item = result.get(str(i + 1))
                if not classification_item:
                    logger.warning("[Classifier] Entry %d missing from LLM result", i + 1)
                    continue

                llm_type = self._normalize_type(classification_item.error_type)
                keywords = classification_item.keywords

                # If LLM classified as NonError, mark it and skip learning
                if llm_type == "NonError":
                    entry["error_type"] = "NonError"
                    entry["is_duplicate"] = False
                    entry["type_source"] = "llm_non_error"
                    logger.debug("[Classifier] Classified as NonError: '%s...'", entry["primary_error"][:60])
                    continue

                logger.debug("[Classifier] Applying: %s to '%s...'", llm_type, entry["primary_error"][:60])

                # Learn validated keywords into store (only for real errors)
                valid_kws = self._validate_keywords(keywords)
                if valid_kws:
                    self._learn_keywords(llm_type, valid_kws)
                    logger.debug("[Classifier] Learned %d keyword(s) for %s: %s", len(valid_kws), llm_type, valid_kws)

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

        # Fan the representative's classification out to its deduped duplicates.
        # Each duplicate keeps its OWN timestamp; only the label is copied.
        if self.dedup:
            for group in dup_groups.values():
                if len(group) < 2:
                    continue
                rep = group[0]
                for dup in group[1:]:
                    dup["error_type"] = rep["error_type"]
                    dup["type_source"] = rep["type_source"]
                    dup["is_duplicate"] = True

        return error_entries

    def get_keyword_store(self) -> Dict[str, List[str]]:
        """Return current keyword store (for inspection/debugging)."""
        return dict(self._keyword_store)

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — dedup + persistence
    # ─────────────────────────────────────────────────────────────

    def _dedup_key(self, primary_error: str) -> str:
        """Normalized key used to collapse identical unknown lines before the LLM."""
        key = re.sub(r"\s+", " ", primary_error.strip().lower())
        if self.dedup_strip_ids:
            key = self._VOLATILE_RE.sub("<id>", key)
        return key

    def _load_keyword_store(self) -> None:
        """Load the learned keyword store from cache_path (best-effort, never crash)."""
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                data = json.load(f)
            store = data.get("store") if isinstance(data, dict) else None
            if isinstance(store, dict) and all(isinstance(v, list) for v in store.values()):
                self._keyword_store = {str(k): [str(x) for x in v] for k, v in store.items()}
                logger.info(
                    "[Classifier] Loaded %d learned error type(s) from %s",
                    len(self._keyword_store), self.cache_path,
                )
            else:
                logger.warning("[Classifier] Ignoring malformed keyword cache at %s", self.cache_path)
        except Exception as e:
            logger.warning("[Classifier] Could not load keyword cache (%s); starting empty", e)

    def save(self) -> None:
        """Persist the keyword store to cache_path atomically, if anything was learned."""
        if not self.cache_path or not self._dirty:
            return
        try:
            target_dir = os.path.dirname(os.path.abspath(self.cache_path))
            os.makedirs(target_dir, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=target_dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "store": self._keyword_store}, f, indent=2)
            os.replace(tmp, self.cache_path)
            self._dirty = False
            logger.debug("[Classifier] Saved keyword store to %s", self.cache_path)
        except Exception as e:
            logger.warning("[Classifier] Could not save keyword cache: %s", e)

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
        best_type = ranked[0][0]

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
            self._dirty = True
            logger.debug("[Classifier] Stored unique keywords: %s", learned)
        else:
            logger.debug("[Classifier] No unique keywords learned (too similar to existing types)")

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — batch sizing
    # ─────────────────────────────────────────────────────────────

    def _compute_batch_size(self) -> int:
        model = self.factory.get_model() if self.factory else "default"
        # An explicit token budget overrides the per-model context window.
        context = self.max_context_tokens or MODEL_CONTEXT_WINDOWS.get(model, 8192)
        usable = max(1, context - PROMPT_OVERHEAD_TOKENS)
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
            # Truncate only the COPY sent to the LLM; never mutate the stored
            # entry (it is reused for clustering and output).
            primary = entry["primary_error"]
            if self.max_entry_chars and len(primary) > self.max_entry_chars:
                primary = primary[: self.max_entry_chars] + "…"
            line = f"[{i + 1}] {primary}"
            # If this entry is a tie, give LLM the candidates as a hint
            candidates = entry.get("_tie_candidates")
            if candidates:
                line += f"  [AMBIGUOUS — could be: {' or '.join(candidates)}]"
            lines.append(line)

        entries_text = "\n".join(lines)

        # Scrub PII/secrets right before the text leaves for the LLM.
        if self.redactor:
            entries_text = self.redactor(entries_text)

        try:
            chain = self._get_structured_llm()
            result = chain.invoke({"entries_text": entries_text})

            # LLM may return classifications keyed by ID (INC2000003) or index (1, 2, 3)
            # Remap IDs to numeric indices for consistency
            classifications = result.classifications

            # Check if keys are numeric strings
            numeric_keys = bool(classifications) and all(k.isdigit() for k in classifications.keys())

            if numeric_keys:
                return classifications

            # Keys are not numeric (e.g. INC IDs). The prompt asks the model to
            # preserve input order, so map the values back to positional indices
            # 1..N by iterating in the order the model returned them. Previously
            # this assigned the FIRST classification to every entry — a bug that
            # gave all entries in the batch the same error type.
            remapped: Dict[str, EntryClassification] = {}
            values = list(classifications.values())
            for orig_idx, _entry in enumerate(batch, 1):
                pos = orig_idx - 1
                if pos < len(values):
                    remapped[str(orig_idx)] = values[pos]
                else:
                    logger.warning(
                        "[Classifier] Entry %d missing from LLM result (keys: %s)",
                        orig_idx,
                        list(classifications.keys()),
                    )
            return remapped if remapped else classifications

        except Exception as e:
            # Recoverable: unknown entries keep UnknownError / first tie candidate.
            # Log at warning without a full traceback (traceback only at debug).
            logger.warning("[Classifier] LLM call failed, falling back: %s", e)
            logger.debug("[Classifier] LLM call traceback", exc_info=True)
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
