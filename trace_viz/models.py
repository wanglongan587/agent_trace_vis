"""Typed data models shared across all parsers and views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation with its input, output, and computed metrics."""

    name: str
    input: dict[str, Any]
    output: str
    is_error: bool
    turn_no: int
    call_idx: int           # 0-based sequential index across the session

    # Computed fields (populated by parsers)
    tiktoken_tokens: int = 0
    output_chars: int = 0
    duration_ms: float = 0.0
    file_path: str = ""

    # Opencode-specific: weight-distributed token cost
    allotted_tokens: int = 0


@dataclass
class Turn:
    """A single LLM inference call with token usage."""

    turn_no: int
    input_tokens: int       # total context window sent (cumulative)
    output_tokens: int

    cache_read: int = 0
    cache_creation: int = 0
    stop_reason: str = ""
    text_content: str = ""
    tool_count: int = 0
    model: str = ""

    # Opencode-specific
    reasoning_tokens: int = 0


@dataclass
class SessionInfo:
    """Metadata about the agent session."""

    model: str = ""
    session_id: str = ""
    tools_available: list[str] = field(default_factory=list)
    title: str = ""
    permission_mode: str = ""


@dataclass
class ResultInfo:
    """Aggregated session outcome and totals."""

    duration_ms: int = 0
    duration_api_ms: int = 0
    num_turns: int = 0
    total_cost_usd: float = 0.0
    is_error: bool = False
    result_text: str = ""

    total_input: int = 0
    total_output: int = 0
    total_cache_creation: int = 0
    total_cache_read: int = 0


@dataclass
class ParseResult:
    """Fully parsed session data returned by every parser."""

    source: str                         # "opencode" | "gemini" | "claude_code"
    raw_events: list[dict[str, Any]]
    session_info: SessionInfo
    result_info: ResultInfo
    turns: list[Turn]
    tool_calls: list[ToolCall]

    parse_errors: int = 0
    parse_debug: dict[str, Any] = field(default_factory=dict)
    subagents: list[dict[str, Any]] = field(default_factory=list)

    # ── Convenience properties ─────────────────────────────────
    @property
    def peak_input_tokens(self) -> int:
        if self.result_info.total_input:
            return self.result_info.total_input
        return max((t.input_tokens for t in self.turns), default=0)

    @property
    def total_cost_usd(self) -> float:
        return self.result_info.total_cost_usd

    @property
    def duration_s(self) -> float:
        return self.result_info.duration_ms / 1000

    @staticmethod
    def empty(source: str) -> "ParseResult":
        return ParseResult(
            source=source,
            raw_events=[],
            session_info=SessionInfo(),
            result_info=ResultInfo(),
            turns=[],
            tool_calls=[],
        )
