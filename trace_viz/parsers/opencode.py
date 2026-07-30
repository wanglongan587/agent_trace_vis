"""Parser for Opencode trace-logger `.ndjson` files.

Key algorithm: weight-based token allotment.
  When multiple tools run in the same globalStep, the input-token delta
  for that step is distributed among them proportionally by their
  tiktoken output size.
"""

from __future__ import annotations

import json
import re
from collections import deque
from typing import Any

import streamlit as st

from trace_viz.models import ParseResult, ResultInfo, SessionInfo, ToolCall, Turn
from trace_viz.utils import count_tokens, to_str


@st.cache_data(show_spinner=False)
def parse(content: bytes) -> ParseResult:
    """Parse Opencode NDJSON content and return a structured ParseResult."""
    raw_events = _load_ndjson(content)
    if not raw_events:
        return ParseResult.empty("opencode")

    session_info = _extract_session_info(raw_events)
    turns = _extract_turns(raw_events)
    tool_calls = _extract_tool_calls(raw_events, turns)
    result_info = _build_result_info(raw_events, turns)
    subagents = _extract_subagents(raw_events)

    return ParseResult(
        source="opencode",
        raw_events=raw_events,
        session_info=session_info,
        result_info=result_info,
        turns=turns,
        tool_calls=tool_calls,
        subagents=subagents,
    )


# ── Private helpers ────────────────────────────────────────────

def _load_ndjson(content: bytes) -> list[dict[str, Any]]:
    events: list[dict] = []
    for line in content.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def _extract_session_info(events: list[dict]) -> SessionInfo:
    for evt in events:
        if evt.get("type") == "session.start":
            return SessionInfo(
                model=evt.get("model", ""),
                session_id=evt.get("sessionID", ""),
                title=evt.get("title", ""),
            )
    return SessionInfo()


def _extract_turns(events: list[dict]) -> list[Turn]:
    turns: list[Turn] = []
    for evt in events:
        if evt.get("type") != "step.finish":
            continue
        turns.append(Turn(
            turn_no=evt["globalStep"],
            input_tokens=evt["cumTokens"]["input"],
            output_tokens=evt["cumTokens"]["output"],
            reasoning_tokens=evt["tokens"].get("reasoning", 0),
            cache_read=evt["tokens"].get("cacheRead", 0),
            cache_creation=evt["tokens"].get("cacheWrite", 0),
            stop_reason=evt.get("reason", ""),
        ))
    return turns


def _extract_tool_calls(
    events: list[dict], turns: list[Turn]
) -> list[ToolCall]:
    # Build lookup maps
    step_map: dict[int, Turn] = {t.turn_no: t for t in turns}
    tool_start_map: dict[str, dict] = {
        e["toolCallId"]: e for e in events if e.get("type") == "tool.start"
    }
    finishes = [e for e in events if e.get("type") == "tool.finish"]

    # Pre-compute tiktoken tokens per finish event (needed for weight calculation)
    finish_tokens: list[int] = [
        count_tokens(to_str(e.get("output", ""))) for e in finishes
    ]

    # Group finish indices by globalStep for parallel-tool weighting
    step_to_indices: dict[int, list[int]] = {}
    for idx, evt in enumerate(finishes):
        gs = evt.get("globalStep", 0)
        step_to_indices.setdefault(gs, []).append(idx)

    tool_calls: list[ToolCall] = []
    for idx, tf in enumerate(finishes):
        gs = tf.get("globalStep", 0)
        cid = tf.get("toolCallId", "")

        # Token delta: how much the context window grew after this step
        curr = step_map.get(gs)
        nxt = step_map.get(gs + 1)
        token_delta = max(0, nxt.input_tokens - curr.input_tokens) if curr and nxt else 0

        tok = finish_tokens[idx]
        parallel_indices = step_to_indices.get(gs, [idx])
        total_parallel_tok = sum(finish_tokens[i] for i in parallel_indices)
        weight = (tok / total_parallel_tok) if total_parallel_tok > 0 else (
            1.0 / len(parallel_indices)
        )

        output_text = to_str(tf.get("output", ""))
        tool_calls.append(ToolCall(
            name=tf.get("tool", ""),
            input=tf.get("args", {}),
            output=output_text,
            is_error=bool(tf.get("isError", False)),
            turn_no=gs,
            call_idx=idx,
            tiktoken_tokens=tok,
            output_chars=tf.get("outputSize", 0) or len(output_text),
            duration_ms=tf.get("duration", 0) or 0,
            allotted_tokens=round(token_delta * weight),
        ))

    return tool_calls


# 真实 opencode 并不会发出独立的 subagent.spawn 事件（插件里监听
# message.part.updated 的 agent/subtask part 分支在实际运行中从未触发过）。
# 子代理的派发在真实 trace 里只是一次普通的 task 工具调用：
#   tool.start / tool.finish，tool == "task"
# 子会话 ID 只出现在 tool.finish.output 的自由文本里，形如：
#   <task id="ses_xxx" state="completed">...
# 需要正则提取。
_TASK_RESULT_RE = re.compile(r'<task\s+id="([^"]+)"\s+state="([^"]*)"')


def _extract_subagents(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从 task 工具调用中还原子代理派发信息，兼容旧的 subagent.spawn 事件。

    注意：真实数据里 tool.start / tool.finish 的 toolCallId 经常对不上——插件在
    缺少显式 ID 时用 `Date.now()` 兜底生成，start 和 finish 取到的是两个不同
    时刻，因此不能按 ID 配对 task 的起止。这里改用 FIFO 顺序配对：task 调用
    绝大多数场景下是"派发-阻塞等待完成"的顺序模式，不会大量并发交错，顺序
    配对足够可靠；配对结果仅用于估算父侧观测到的派发耗时，不影响子会话 ID
    的提取（那部分始终来自 finish.output 的正则匹配）。
    """
    pending_starts: deque[dict[str, Any]] = deque()
    subagents: list[dict[str, Any]] = []
    seen_child_ids: set[str] = set()

    for evt in events:
        etype = evt.get("type")
        if etype == "tool.start" and evt.get("tool") == "task":
            pending_starts.append(evt)
            continue
        if etype != "tool.finish" or evt.get("tool") != "task":
            continue

        start_evt = pending_starts.popleft() if pending_starts else None
        args = evt.get("args") or {}
        output = to_str(evt.get("output", ""))
        m = _TASK_RESULT_RE.search(output)
        child_id = m.group(1) if m else ""
        state = m.group(2) if m else ("error" if evt.get("isError") else "unknown")

        duration_ms = None
        if start_evt is not None:
            duration_ms = max(0, evt.get("ts", 0) - start_evt.get("ts", 0))

        if child_id:
            seen_child_ids.add(child_id)
        subagents.append({
            "childSessionID": child_id,
            "agentName": str(args.get("subagent_type", "")),
            "description": str(args.get("description", "")),
            "state": state,
            "globalStep": evt.get("globalStep", 0),
            "ts": evt.get("ts", 0),
            "dispatchDurationMs": duration_ms,
        })

    # 还在进行中的 task 调用（只有 start，没有 finish）：拿不到子会话 ID
    # （ID 只出现在 finish 的输出里），但仍应让用户知道"有一次派发还未完成"，
    # 而不是让它凭空消失。
    for start_evt in pending_starts:
        args = start_evt.get("args") or {}
        subagents.append({
            "childSessionID": "",
            "agentName": str(args.get("subagent_type", "")),
            "description": str(args.get("description", "")),
            "state": "running",
            "globalStep": start_evt.get("globalStep", 0),
            "ts": start_evt.get("ts", 0),
            "dispatchDurationMs": None,
        })

    # 向后兼容：如果某个环境的插件确实发出了 subagent.spawn（例如未来版本
    # 修好了 message.part.updated 的那个分支），也一并纳入，按 childSessionID
    # 去重，避免同一个子会话被展示两次。
    for evt in events:
        if evt.get("type") != "subagent.spawn":
            continue
        cid = evt.get("childSessionID", "")
        if cid and cid in seen_child_ids:
            continue
        if cid:
            seen_child_ids.add(cid)
        subagents.append({
            "childSessionID": cid,
            "agentName": evt.get("agentName", ""),
            "description": "",
            "state": "unknown",
            "globalStep": evt.get("globalStep", 0),
            "ts": evt.get("ts", 0),
            "dispatchDurationMs": None,
        })

    return subagents


def _build_result_info(events: list[dict], turns: list[Turn]) -> ResultInfo:
    info = ResultInfo(num_turns=len(turns))

    for evt in events:
        if evt.get("type") == "session.end":
            ti = evt.get("totalTokens", {})
            info.total_input = ti.get("input", 0)
            info.total_output = ti.get("output", 0)
            break

    if not info.total_input and turns:
        info.total_input = turns[-1].input_tokens
        info.total_output = turns[-1].output_tokens

    if len(events) >= 2:
        info.duration_ms = events[-1]["ts"] - events[0]["ts"]

    return info
