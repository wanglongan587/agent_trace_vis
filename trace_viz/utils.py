"""Shared utilities: token counting, text sanitization, formatting."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import tiktoken
import tiktoken.load

# ── Tiktoken encoder (loaded from local cache if available) ────
_CACHE_PATH = os.path.expanduser("~/.cache/tiktoken/cl100k_base.tiktoken")
_ENCODER: tiktoken.Encoding | None = None

if os.path.exists(_CACHE_PATH):
    try:
        _ranks = tiktoken.load.load_tiktoken_bpe(_CACHE_PATH)
        _ENCODER = tiktoken.Encoding(
            name="cl100k_base",
            pat_str=(
                r"(?i:[sdmt]|ll|ve|re)|[^\n\pL\pP\p{Nd}\p{Sc}\p{Sk}\p{So}]+"
                r"|\p{Nd}+|\p{Sc}[\p{Nd}]*|\p{Sc}+|\p{Sk}+|\p{So}+"
                r"|\p{Ll}\p{Lo}|\p{L}\p{L}\p{L}*\p{Ll}\p{Lo}"
            ),
            mergeable_ranks=_ranks,
            special_tokens={
                "<|endoftext|>": 100257,
                "<|fim_prefix|>": 100258,
                "<|fim_middle|>": 100259,
                "<|fim_suffix|>": 100260,
                "<|endofprompt|>": 100276,
            },
        )
    except Exception:
        pass


@lru_cache(maxsize=4096)
def count_tokens(text: str) -> int:
    """Approximate token count via cl100k_base; falls back to len/4."""
    if not text:
        return 0
    if _ENCODER is not None:
        return len(_ENCODER.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)


# ── Text helpers ───────────────────────────────────────────────

def to_str(value: Any) -> str:
    """Coerce any value (bytes, dict, list, None) to a plain string."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


_MERMAID_TABLE = str.maketrans({
    # ASCII straight quotes collapse to single quotes so they cannot prematurely close
    # the outer double-quoted label Mermaid wraps the sanitized text in.
    '"': "'",
    ":": "：", "\n": " ", "[": "(", "]": ")",
    ";": ",", "#": "", "<": "＜", ">": "＞", "&": "＆",
    # Curly/smart/fullwidth quotes are NOT caught by the ASCII mapping above; left
    # unescaped they terminate Mermaid's double-quoted label mid-string and surface
    # as "Syntax error in text" in the sequence diagram. Normalize all of them to
    # ASCII single quotes so agent-generated prose (which freely uses "smart quotes")
    # renders safely inside a Mermaid message label.
    "\u201c": "'",  # left double quotation mark "
    "\u201d": "'",  # right double quotation mark "
    "\u201e": "'",  # double low-9 quotation mark „
    "\u201f": "'",  # double high-reversed-9 quotation mark ‟
    "\uff02": "'",  # fullwidth quotation mark ＂
    "\u2018": "'",  # left single quotation mark '
    "\u2019": "'",  # right single quotation mark '
    "\u201a": "'",  # single low-9 quotation mark ‚
    "\u201b": "'",  # single high-reversed-9 quotation mark ‛
})


def sanitize_mermaid(text: str, max_len: int = 50) -> str:
    """Make a string safe for use in a Mermaid diagram label."""
    text = to_str(text).strip().translate(_MERMAID_TABLE)
    return (text[:max_len] + "…") if len(text) > max_len else text


def mermaid_quote(text: Any) -> str:
    """Return a Mermaid-safe double-quoted label."""
    return f'"{sanitize_mermaid(to_str(text))}"'


def format_duration(ms: int | float) -> str:
    """Human-readable duration from milliseconds."""
    if not ms:
        return "—"
    s = ms / 1000
    if s < 60:
        return f"{s:.1f}s"
    m, rem = divmod(int(s), 60)
    return f"{m}m {rem}s"


def decode_bytes(content: bytes) -> str | None:
    """Try common encodings in order; return None if all fail."""
    for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            return content.decode(enc)
        except (UnicodeDecodeError, Exception):
            continue
    return None
