"""
redaction.py
============
Best-effort PII / secret scrubbing applied ONLY to text that is about to leave
the machine for a cloud LLM. Local providers (Ollama) need no scrubbing, so
redaction is opt-in / auto-disabled for them (see LogTracer `redact` arg).

`redact(text)` replaces emails, IPs, bearer tokens, API keys, JWTs and long
hex secrets with stable placeholders. It is intentionally conservative — it
masks high-confidence secret shapes rather than aggressively nuking every long
token (which would garble the very log lines we want the model to reason over).
"""

import re
from typing import List, Tuple

# (compiled pattern, replacement). Order matters: more specific shapes first.
_RULES: List[Tuple["re.Pattern[str]", str]] = [
    # Bearer tokens — keep the keyword, mask the credential.
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"), "Bearer <redacted-token>"),
    # JWTs (three base64url segments).
    (re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"), "<redacted-jwt>"),
    # OpenAI-style and AWS access keys.
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), "<redacted-api-key>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<redacted-aws-key>"),
    # key=value / token: value secrets.
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|pwd)\b\s*[=:]\s*\S+"),
     r"\1=<redacted>"),
    # Emails.
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "<redacted-email>"),
    # IPv4 addresses.
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<redacted-ip>"),
    # IPv6 addresses (loose).
    (re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}\b"), "<redacted-ip>"),
    # Long lowercase hex blobs (>=32) — hashes / secret material.
    (re.compile(r"\b[0-9a-f]{32,}\b"), "<redacted-hex>"),
]


def redact(text: str) -> str:
    """Return `text` with high-confidence PII / secrets masked by placeholders."""
    if not text:
        return text
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text
