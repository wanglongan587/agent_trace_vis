"""Claude Code visualization view.

Supports two data sources:
  1. stream-json file  — `claude -p "task" --output-format stream-json > trace.ndjson`
  2. transcript JSONL  — auto-saved to ~/.claude/projects/<hash>/<session>.jsonl
                         during every interactive terminal session (no setup needed)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trace_viz.config import SAFE_PALETTE
from trace_viz.models import ParseResult
from trace_viz.parsers.claude_code import parse
from trace_viz.utils import format_duration, mermaid_quote, sanitize_mermaid, to_str
from trace_viz.views.shared import (
    mermaid_controls,
    raw_events_tab,
    render_mermaid,
    sample_events,
    token_delta_fig,
    token_trend_fig,
    tool_efficiency_table,
    tool_inspector,
    tool_success_rate,
    tool_tiktoken_fig,
)

# ── Transcript root ────────────────────────────────────────────
_TRANSCRIPT_ROOT = Path.home() / ".claude" / "projects"


def render() -> None:
    """Standalone entry point: picks a data source via the sidebar, then renders it."""
    result = _sidebar()
    if result is None:
        _show_quickstart()
        return
    render_body(result)


def render_body(result: ParseResult) -> None:
    """Renders an already-parsed result, shared by the standalone and embedded flows."""
    df_tools = _build_tools_df(result)

    _sidebar_meta(result)
    _metrics_row(result, df_tools)
    st.markdown("---")

    is_transcript = result.parse_debug.get("format") == "transcript"
    df_turns = pd.DataFrame([t.__dict__ for t in result.turns]) if result.turns else pd.DataFrame()

    tabs = ["Token 趋势", "工具执行",  "成本分析", "原始数据"]
    if is_transcript:
        tabs.insert(1, "时间轴")   # extra tab only available with real timestamps

    tab_objects = st.tabs(tabs)
    idx = 0

    with tab_objects[idx]: _tab_tokens(df_turns);                  idx += 1
    if is_transcript:
        with tab_objects[idx]: _tab_timeline(result);              idx += 1
    with tab_objects[idx]: _tab_tools(df_tools);                   idx += 1
    with tab_objects[idx]: _tab_cost(result, df_turns);            idx += 1
    with tab_objects[idx]: _tab_raw(result);                       idx += 1

    # Deep-dive outside tabs
    if not df_tools.empty:
        st.markdown("---")
        st.subheader("单个工具深度诊断")
        tool_inspector(df_tools)


# ── Sidebar ────────────────────────────────────────────────────

def _sidebar() -> ParseResult | None:
    with st.sidebar:
        st.markdown("### 数据来源")
        mode = st.radio(
            "选择方式",
            ["交互会话记录（~/.claude）", "上传文件"],
            key="cc_src_mode",
        )

        if mode == "交互会话记录（~/.claude）":
            return _sidebar_transcript()
        else:
            return _sidebar_upload()


def _sidebar_transcript() -> ParseResult | None:
    """Browse ~/.claude/projects/ and let the user pick a session."""
    with st.sidebar:
        custom_root = st.text_input(
            "transcript 目录",
            str(_TRANSCRIPT_ROOT),
            key="cc_root",
        )
        root = Path(custom_root)

        if not root.exists():
            st.warning(f"目录不存在：`{root}`")
            return None

        # Collect all JSONL files, sorted newest-first
        all_files = sorted(
            root.rglob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not all_files:
            st.info("未找到 `.jsonl` 文件，请确认路径。")
            return None

        # Build display labels: "<project-dir> / <filename>  (date)"
        def _label(p: Path) -> str:
            try:
                rel   = p.relative_to(root)
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
                size  = p.stat().st_size
                return f"{rel.parent.name[:20]}/{p.stem[:18]}  {mtime:%m-%d %H:%M}  {size//1024}KB"
            except Exception:
                return str(p)

        labels      = [_label(p) for p in all_files]
        chosen_label = st.selectbox(
            f"选择会话（共 {len(all_files)} 个）",
            labels,
            key="cc_session_sel",
        )
        chosen_path = all_files[labels.index(chosen_label)]
        st.caption(str(chosen_path))

        if st.button("加载此会话", type="primary", use_container_width=True):
            st.session_state["cc_content"] = chosen_path.read_bytes()
            st.session_state.pop("cc_result", None)
            st.rerun()

        content = st.session_state.get("cc_content")
        if content is None:
            return None

        if "cc_result" not in st.session_state:
            with st.spinner("解析中…"):
                st.session_state["cc_result"] = parse(content)

        return st.session_state.get("cc_result")


def _sidebar_upload() -> ParseResult | None:
    """Accept a manually uploaded JSONL / NDJSON file."""
    with st.sidebar:
        uploaded = st.file_uploader(
            "上传日志文件",
            type=["ndjson", "jsonl", "txt", "json"],
            key="cc_upload",
        )
        st.divider()
        st.markdown("**stream-json 模式生成方法**")
        st.code(
            "claude --output-format stream-json \\\n"
            "  -p \"你的任务\" > trace.ndjson",
            language="bash",
        )

    if uploaded is None:
        return None
    return parse(uploaded.getvalue())


def _sidebar_meta(result: ParseResult) -> None:
    with st.sidebar:
        st.divider()
        st.markdown("### 会话信息")
        si  = result.session_info
        dbg = result.parse_debug
        st.text(f"格式: {dbg.get('format', '?')}")
        if si.model:      st.text(f"模型: {si.model}")
        if si.session_id: st.text(f"Session: {si.session_id[:20]}…")
        if si.title:      st.text(f"目录: …{si.title[-30:]}")
        st.text(f"LLM 轮次: {len(result.turns)}")
        st.text(f"工具调用: {len(result.tool_calls)}")
        if result.total_cost_usd:
            st.text(f"总费用: ${result.total_cost_usd:.4f}")
        if dbg.get("version"):
            st.caption(f"claude v{dbg['version']}")


# ── Metrics row ────────────────────────────────────────────────

def _metrics_row(result: ParseResult, df_tools: pd.DataFrame) -> None:
    ri = result.result_info
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("LLM 推理轮次",     len(result.turns))
    m2.metric("工具调用总数",      len(result.tool_calls))
    m3.metric("峰值 Input Tokens", f"{result.peak_input_tokens:,}" if result.peak_input_tokens else "—")
    m4.metric("总耗时",            format_duration(ri.duration_ms))
    m5.metric("总费用",            f"${result.total_cost_usd:.4f}" if result.total_cost_usd else "—")
    m6.metric("工具调用成功率",     f"{tool_success_rate(df_tools):.1f}%")


# ── DataFrame builders ─────────────────────────────────────────

def _build_tools_df(result: ParseResult) -> pd.DataFrame:
    if not result.tool_calls:
        return pd.DataFrame()
    rows = []
    for tc in result.tool_calls:
        d = tc.__dict__.copy()
        d["_input_dict"] = tc.input
        d["input"]       = json.dumps(tc.input, ensure_ascii=False, indent=2)
        rows.append(d)
    return pd.DataFrame(rows)


def _build_timeline_df(result: ParseResult) -> pd.DataFrame:
    """Extract timestamp-bearing events into a flat DataFrame for the timeline tab.

    Tool results are nested inside user messages (same as stream-json), so we
    emit a separate synthetic "tool_result" row for each one found.
    """
    rows = []
    for evt in result.raw_events:
        ts_raw = evt.get("timestamp")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except Exception:
            continue
        etype = evt.get("type", "")

        if etype == "assistant":
            usage = (evt.get("message") or {}).get("usage") or {}
            in_t  = usage.get("input_tokens", "")
            out_t = usage.get("output_tokens", "")
            rows.append({
                "timestamp": ts,
                "type":      "assistant",
                "label":     f"assistant  in={in_t} out={out_t}",
            })

        elif etype == "user":
            msg         = (evt.get("message") or {})
            raw_content = msg.get("content", [])
            if isinstance(raw_content, str):
                # Plain-text user turn
                rows.append({
                    "timestamp": ts,
                    "type":      "user",
                    "label":     f"user  {raw_content[:40]}",
                })
            elif isinstance(raw_content, list):
                has_tool_result = False
                has_text        = False
                for block in raw_content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "tool_result":
                        has_tool_result = True
                        tid = block.get("tool_use_id") or block.get("toolUseId") or ""
                        err = " [ERROR]" if block.get("is_error") else ""
                        rows.append({
                            "timestamp": ts,
                            "type":      "tool_result",
                            "label":     f"tool_result{err}  id={tid[:14]}",
                        })
                    elif btype == "text":
                        has_text = True
                        txt      = block.get("text", "")[:40]
                if has_text and not has_tool_result:
                    rows.append({
                        "timestamp": ts,
                        "type":      "user",
                        "label":     f"user  {txt}",
                    })

    return pd.DataFrame(rows)


# ── Tab 2: Token trends ────────────────────────────────────────

def _tab_tokens(df_turns: pd.DataFrame) -> None:
    if df_turns.empty:
        st.info("暂无 Token 数据（未找到 assistant 事件）")
        return

    st.subheader("每轮 Token 用量（上下文窗口增长曲线）")
    st.plotly_chart(token_trend_fig(df_turns), use_container_width=True)

    st.divider()
    st.subheader("Input 增量（每轮上下文新增量）")
    st.plotly_chart(token_delta_fig(df_turns), use_container_width=True)

    if "cache_read" in df_turns.columns and df_turns["cache_read"].sum() > 0:
        st.divider()
        st.subheader("缓存命中率（Cache Read / Input）")
        df_c = df_turns.copy()
        df_c["cache_ratio"] = (
            df_c["cache_read"] / df_c["input_tokens"].replace(0, 1) * 100
        ).round(1)
        fig_cache = go.Figure(go.Bar(
            x=df_c["turn_no"], y=df_c["cache_ratio"],
            marker_color="#14b8a6",
            text=df_c["cache_ratio"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))
        fig_cache.update_layout(height=280, xaxis_title="Turn",
                                yaxis_title="Cache Hit %",
                                margin=dict(t=20, b=0), showlegend=False)
        st.plotly_chart(fig_cache, use_container_width=True)


# ── Tab 3: Timeline (transcript only) ─────────────────────────

def _tab_timeline(result: ParseResult) -> None:
    st.subheader("消息时间轴（真实时间戳）")
    df_tl = _build_timeline_df(result)
    if df_tl.empty:
        st.info("未找到带时间戳的事件")
        return

    # ── Scatter: all messages on a timeline ───────────────────
    type_color = {
        "user":        "#64748b",
        "assistant":   "#1a73e8",
        "tool_result": "#34a853",
    }
    fig_scatter = px.scatter(
        df_tl, x="timestamp", y="type",
        color="type", color_discrete_map=type_color,
        hover_data={"label": True, "timestamp": False},
        labels={"type": "事件类型", "timestamp": "时间"},
    )
    fig_scatter.update_traces(marker=dict(size=10, opacity=0.8))
    fig_scatter.update_layout(height=240, showlegend=False, margin=dict(t=10, b=0))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ── Per-turn response latency ──────────────────────────────
    st.subheader("LLM 响应延迟（用户发送 → 收到回复）")
    user_ts = df_tl[df_tl["type"] == "user"]["timestamp"].sort_values().tolist()
    asst_ts = df_tl[df_tl["type"] == "assistant"]["timestamp"].sort_values().tolist()

    latency_rows = []
    for i, at in enumerate(asst_ts):
        # Find the last user message before this assistant message
        prior = [ut for ut in user_ts if ut < at]
        if prior:
            delta_s = (at - max(prior)).total_seconds()
            latency_rows.append({"turn": i + 1, "延迟(s)": round(delta_s, 2)})

    if latency_rows:
        df_lat = pd.DataFrame(latency_rows)
        fig_lat = go.Figure(go.Bar(
            x=df_lat["turn"], y=df_lat["延迟(s)"],
            marker_color="#1a73e8",
            text=df_lat["延迟(s)"].apply(lambda v: f"{v:.1f}s"),
            textposition="outside",
        ))
        fig_lat.update_layout(
            height=280, xaxis_title="LLM 推理轮次", yaxis_title="秒",
            margin=dict(t=20, b=0), showlegend=False,
        )
        st.plotly_chart(fig_lat, use_container_width=True)
    else:
        st.info("轮次过少，无法计算延迟")

    st.divider()

    # ── Full event log ─────────────────────────────────────────
    st.subheader("事件时间明细")
    st.dataframe(
        df_tl.assign(timestamp=df_tl["timestamp"].dt.strftime("%H:%M:%S.%f").str[:-3]),
        use_container_width=True, height=300,
    )


# ── Tab 4: Tool execution ──────────────────────────────────────

def _tab_tools(df_tools: pd.DataFrame) -> None:
    if df_tools.empty:
        st.info("暂无工具调用记录")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("各工具调用次数")
        tc = df_tools["name"].value_counts().reset_index()
        tc.columns = ["工具名称", "次数"]
        fa = px.bar(tc, x="工具名称", y="次数", color="工具名称", text="次数",
                    color_discrete_sequence=SAFE_PALETTE)
        fa.update_traces(textposition="outside")
        fa.update_layout(height=320, margin=dict(t=10, b=0), showlegend=False)
        st.plotly_chart(fa, use_container_width=True)

    with col_b:
        st.subheader("每次调用的输出大小（Tiktoken Tokens）")
        st.plotly_chart(tool_tiktoken_fig(df_tools), use_container_width=True)

    st.divider()
    st.subheader("工具效率汇总")
    tool_efficiency_table(df_tools)

# ── Tab 6: Cost analysis ───────────────────────────────────────

def _tab_cost(result: ParseResult, df_turns: pd.DataFrame) -> None:
    st.subheader("Token 构成与费用拆解")
    ri = result.result_info

    if not any([ri.total_input, ri.total_output, ri.total_cost_usd]):
        st.info("当前格式无总费用信息（transcript 模式不含 cost，需通过 Anthropic 控制台查看）")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Token 构成")
            labels, vals, colors = [], [], []
            for lbl, v, color in [
                ("Input（非缓存）", ri.total_input - ri.total_cache_read, "#1a73e8"),
                ("Cache Read",      ri.total_cache_read,                  "#14b8a6"),
                ("Cache Creation",  ri.total_cache_creation,              "#a855f7"),
                ("Output",          ri.total_output,                      "#34a853"),
            ]:
                if v and v > 0:
                    labels.append(lbl); vals.append(v); colors.append(color)
            if vals:
                fig_pie = px.pie(
                    pd.DataFrame({"类型": labels, "数量": vals}),
                    names="类型", values="数量", hole=0.4,
                    color_discrete_sequence=colors,
                )
                fig_pie.update_traces(textinfo="label+percent+value")
                fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0))
                st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            st.subheader("数据明细")
            st.metric("总 Input Tokens",  f"{ri.total_input:,}")
            st.metric("总 Output Tokens", f"{ri.total_output:,}")
            if ri.total_cache_read:     st.metric("Cache Read",     f"{ri.total_cache_read:,}")
            if ri.total_cache_creation: st.metric("Cache Creation", f"{ri.total_cache_creation:,}")
            if ri.total_cost_usd:       st.metric("总费用 (USD)",   f"${ri.total_cost_usd:.6f}")
            if ri.duration_ms:          st.metric("总耗时",          format_duration(ri.duration_ms))
            if ri.duration_api_ms:
                st.metric("API 等待时间", format_duration(ri.duration_api_ms))
                local_ms = ri.duration_ms - ri.duration_api_ms
                if local_ms > 0:
                    st.metric("本地处理时间", format_duration(local_ms))

    if not df_turns.empty:
        st.divider()
        st.subheader("逐轮 Token 明细")
        disp_cols = {
            "turn_no": "Turn", "input_tokens": "Input", "output_tokens": "Output",
            "cache_read": "CacheRead", "cache_creation": "CacheCreation",
            "tool_count": "工具调用数", "stop_reason": "StopReason",
        }
        disp = df_turns[[c for c in disp_cols if c in df_turns.columns]].rename(columns=disp_cols)
        st.dataframe(disp, use_container_width=True)


# ── Tab 7: Raw data ────────────────────────────────────────────

def _tab_raw(result: ParseResult) -> None:
    st.subheader(f"全部事件（{len(result.raw_events):,} 条）")
    raw_events_tab(result.raw_events, key_prefix="cc_raw", type_field="type")


# ── Mermaid builder ────────────────────────────────────────────

def _iter_content(raw_content: object) -> list[dict]:
    """Normalise a content field into a list of dicts.

    In transcript JSONL the content may arrive as:
      - a plain string          → wrap as {"type":"text","text":...}
      - a list of dicts         → return as-is (filter out non-dicts)
      - a list of strings       → wrap each as {"type":"text","text":...}
    """
    if isinstance(raw_content, str):
        return [{"type": "text", "text": raw_content}]
    if not isinstance(raw_content, list):
        return []
    out = []
    for item in raw_content:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            out.append({"type": "text", "text": item})
    return out


def _build_mermaid(events: list[dict], *, is_transcript: bool) -> str:
    lines = [
        "sequenceDiagram", "    autonumber",
        "    participant U as User",
        "    participant A as Claude",
        "    participant T as Tool",
    ]

    for evt in events:
        if not isinstance(evt, dict):
            continue
        etype = evt.get("type", "")
        ts    = ""
        if is_transcript and evt.get("timestamp"):
            try:
                ts = " " + datetime.fromisoformat(
                    evt["timestamp"].replace("Z", "+00:00")
                ).strftime("%H:%M:%S")
            except Exception:
                pass

        if etype == "system":
            model = evt.get("model", "Claude")
            lines.append(f"    Note over U,T: 会话初始化 model={sanitize_mermaid(model, 28)}")

        elif etype == "assistant":
            msg = evt.get("message") or {}
            if not isinstance(msg, dict):
                continue
            usage = msg.get("usage") or {}
            in_t  = usage.get("input_tokens", "")
            out_t = usage.get("output_tokens", "")
            tok_s = f"in={in_t} out={out_t}" if in_t else ""
            lines.append(f"    Note over A: {mermaid_quote('LLM推理 ' + tok_s + ts)}")

            for block in _iter_content(msg.get("content", [])):
                btype = block.get("type", "")
                if btype == "text":
                    txt = sanitize_mermaid(block.get("text", ""), 60)
                    if txt:
                        lines.append(f"    A->>U: {mermaid_quote(txt)}")
                elif btype == "tool_use":
                    name = sanitize_mermaid(block.get("name", "tool"), 25)
                    inp  = block.get("input") or {}
                    hint = sanitize_mermaid(
                        to_str(next(iter(inp.values()), "")) if inp else "", 32
                    )
                    label = name + (f"({hint})" if hint else "")
                    lines.append(f"    A->>+T: {mermaid_quote(label)}")

        elif etype == "user":
            # Both stream-json and transcript: tool_result nested in user message
            msg = evt.get("message") or {}
            if not isinstance(msg, dict):
                continue
            has_tool_result = False
            for block in _iter_content(msg.get("content", [])):
                btype = block.get("type", "")
                if btype == "tool_result":
                    has_tool_result = True
                    raw = block.get("content", "")
                    content_str = "\n".join(
                        (c.get("text", str(c)) if isinstance(c, dict) else str(c))
                        for c in raw
                    ) if isinstance(raw, list) else to_str(raw)
                    err   = " [ERROR]" if block.get("is_error") else ""
                    label = sanitize_mermaid(content_str or "done", 50) + err
                    lines.append(f"    T-->>-A: {mermaid_quote(label + ts)}")
                elif btype == "text" and not has_tool_result:
                    txt = sanitize_mermaid(block.get("text", ""), 60)
                    if txt:
                        lines.append(f"    U->>A: {mermaid_quote(txt + ts)}")

        elif etype == "tool_result":
            # Keep for safety: some future versions may emit top-level tool_result
            content = evt.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    (c.get("text", str(c)) if isinstance(c, dict) else str(c))
                    for c in content
                )
            err   = " [ERROR]" if evt.get("isError") else ""
            label = sanitize_mermaid(to_str(content or "done"), 50) + err
            lines.append(f"    T-->>-A: {mermaid_quote(label + ts)}")

        elif etype == "result":
            parts = ["任务完成"]
            n_t  = evt.get("num_turns", "")
            dur  = evt.get("duration_ms", 0)
            cost = evt.get("total_cost_usd", 0)
            if n_t:  parts.append(f"共{n_t}轮")
            if dur:
                try: parts.append(f"耗时{int(dur) // 1000}s")
                except (TypeError, ValueError): pass
            if cost:
                try: parts.append(f"${float(cost):.4f}")
                except (TypeError, ValueError): pass
            lines.append(f"    A->>U: {mermaid_quote(' '.join(parts))}")

    return "\n".join(lines)


# ── Quickstart hint ────────────────────────────────────────────

def _show_quickstart() -> None:
    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 交互会话模式（推荐）")
        st.markdown(
            "Claude Code 在每次交互会话中会**自动**把完整对话保存为 JSONL 文件，"
            "无需任何额外配置。"
        )
        st.code(
            "~/.claude/projects/<项目hash>/<session-id>.jsonl",
            language="text",
        )
        st.markdown(
            "在左侧切换到 **交互会话记录** 模式，工具会自动扫描该目录，"
            "选择一个会话即可加载。"
        )
        st.info(
            "每行一条消息，包含 `type`、`timestamp`、`uuid`、"
            "`message`（含 usage）等字段。\n\n"
            "transcript 格式额外提供：\n"
            "- 每条消息的真实时间戳\n"
            "- 工具调用的实际耗时（从时间差计算）\n"
            "- LLM 响应延迟分析"
        )

    with col_r:
        st.markdown("#### `-p` 模式（stream-json）")
        st.markdown("适合非交互式单次任务：")
        st.code(
            "claude --output-format stream-json \\\n"
            "  -p \"你的任务描述\" \\\n"
            "  > claude_trace.ndjson",
            language="bash",
        )
        st.markdown("生成后切换左侧到 **上传文件** 模式加载。")
        st.caption("需要 Claude Code ≥ 1.x。运行 `claude --version` 确认。")
