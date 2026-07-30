"""Opencode trace visualization view."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trace_viz.config import OC_COLORS, SAFE_PALETTE
from trace_viz.models import ParseResult
from trace_viz.parsers.opencode import parse
from trace_viz.utils import format_duration, mermaid_quote, sanitize_mermaid, to_str
from trace_viz.views.shared import (
    mermaid_controls,
    render_mermaid,
    sample_events,
    token_delta_fig,
    token_trend_fig,
    tool_efficiency_table,
    tool_inspector,
    tool_success_rate,
    tool_tiktoken_fig,
)

_OC_TRACE_DIR = Path.home() / ".local" / "share" / "opencode" / "trace"


def render() -> None:
    """Standalone entry point: uploads a trace file via the sidebar, then renders it."""
    uploaded = st.sidebar.file_uploader("上传 .ndjson 日志文件", type=["ndjson"])
    if uploaded is None:
        st.info("请在左侧边栏上传由 trace-logger 生成的 `.ndjson` 追踪日志文件。")
        return

    result = parse(uploaded.getvalue())
    if not result.raw_events:
        st.error("未解析到任何事件，请确认文件格式。")
        return

    render_body(result)


def render_body(result: ParseResult) -> None:
    """Renders an already-parsed result, shared by the standalone and embedded flows."""
    if not result.raw_events:
        st.error("未解析到任何事件，请确认文件格式。")
        return

    df_tools = _build_tools_df(result)

    _sidebar_meta(result)
    _metrics_row(result, df_tools)
    st.markdown("---")

    df_turns = pd.DataFrame([t.__dict__ for t in result.turns]) if result.turns else pd.DataFrame()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["总览", "Subagent", "Token 趋势", "工具执行与消耗", "原始数据"]
    )
    with tab1: _tab_overview(result, df_tools)
    with tab2: _tab_subagents(result)
    with tab3: _tab_tokens(df_turns)
    with tab4: _tab_tools(df_tools)
    with tab5: _tab_raw(result, df_turns, df_tools)

    # ── 单个工具深度诊断（位于所有 Tab 之外，页面底部）─────────────
    if not df_tools.empty:
        st.markdown("---")
        st.subheader("单个工具深度诊断")
        tool_inspector(df_tools)


# ── Sidebar ────────────────────────────────────────────────────

def _sidebar_meta(result: ParseResult) -> None:
    with st.sidebar:
        st.markdown("### 会话元数据")
        si = result.session_info
        if si.model:  st.text(f"模型: {si.model}")
        if si.title:  st.text(f"标题: {si.title}")
        st.text(f"总事件数: {len(result.raw_events)}")


# ── Metrics row ────────────────────────────────────────────────

def _metrics_row(result: ParseResult, df_tools: pd.DataFrame) -> None:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("总 Steps（LLM 轮次）", len(result.turns))
    m2.metric("工具调用总数",         len(result.tool_calls))
    m3.metric("峰值 Input Tokens",    f"{result.peak_input_tokens:,}")
    m4.metric("Session 总持续时间",    format_duration(result.result_info.duration_ms))
    m5.metric("工具调用成功率",        f"{tool_success_rate(df_tools):.1f}%")


# ── DataFrame builder ──────────────────────────────────────────

def _build_tools_df(result: ParseResult) -> pd.DataFrame:
    if not result.tool_calls:
        return pd.DataFrame()
    rows = []
    for tc in result.tool_calls:
        d = tc.__dict__.copy()
        d["_input_dict"] = tc.input
        d["input"] = json.dumps(tc.input, ensure_ascii=False, indent=2)
        rows.append(d)
    return pd.DataFrame(rows)


# ── Subagent helpers ────────────────────────────────────────────

def _load_subagent_result(child_session_id: str) -> ParseResult | None:
    """Attempt to resolve and parse a subagent's own trace file, if present locally."""
    if not child_session_id:
        return None
    p = _OC_TRACE_DIR / f"{child_session_id}.ndjson"
    if not p.is_file():
        return None
    try:
        return parse(p.read_bytes())
    except Exception:
        return None


# ── Tab 1: Overview ────────────────────────────────────────────

def _tab_overview(result: ParseResult, df_tools: pd.DataFrame) -> None:
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("事件类型分布")
        types = pd.Series([e.get("type", "") for e in result.raw_events])
        cc = types.value_counts().reset_index()
        cc.columns = ["type", "count"]
        fig = px.pie(cc, names="type", values="count",
                     color="type", color_discrete_map=OC_COLORS, hole=0.4)
        fig.update_traces(textinfo="label+percent+value")
        fig.update_layout(showlegend=False, margin=dict(t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("工具调用分布")
        if not df_tools.empty:
            tc = df_tools["name"].value_counts().reset_index()
            tc.columns = ["工具名称", "次数"]
            fig2 = px.bar(tc, x="次数", y="工具名称", orientation="h",
                          color="工具名称", color_discrete_sequence=SAFE_PALETTE)
            fig2.update_layout(yaxis=dict(autorange="reversed"),
                               margin=dict(t=0, b=0), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("暂无工具调用")

    st.divider()
    st.subheader("主要步骤时序图")
    max_ev, theme, row_h = mermaid_controls(key_prefix="oc_seq")
    units = _build_sequence_units(result.raw_events)
    sampled = sample_events(units, max_ev)
    if not sampled:
        st.warning("未找到可渲染的关键事件")
    else:
        src = _build_mermaid(sampled)
        render_mermaid(src, theme=theme, row_height=row_h, event_count=len(sampled))
        with st.expander("复制 Mermaid 源码"):
            st.code(src, language="text")

    step_starts = [e for e in result.raw_events if e.get("type") == "step.start"]
    if step_starts:
        st.divider()
        st.subheader("模型调用时间")
        df_ts = pd.DataFrame([{
            "Global Step": e.get("globalStep"),
            "时间": pd.to_datetime(e.get("ts", 0), unit="ms"),
        } for e in step_starts])
        fig_ts = px.scatter(df_ts, x="时间", y="Global Step")
        fig_ts.update_traces(marker=dict(size=10, color="#1a73e8"))
        fig_ts.update_layout(height=240, margin=dict(t=10, b=0))
        st.plotly_chart(fig_ts, use_container_width=True)
        st.dataframe(
            df_ts.assign(时间=df_ts["时间"].dt.strftime("%H:%M:%S.%f").str[:-3]),
            use_container_width=True, height=200,
        )


# ── Tab 2: Subagents ────────────────────────────────────────────

# state 取自 task 调用输出里 <task id="..." state="..."> 的 state 属性
# （completed/failed 等），running 是本地补的状态，表示只看到了 tool.start
# 还没等到 tool.finish（还没拿到子会话 ID，因为 ID 只出现在 finish 的输出里）。
_STATE_LABELS = {
    "completed": "✅ 已完成",
    "failed":    "❌ 失败",
    "error":     "❌ 出错",
    "running":   "⏳ 进行中",
    "unknown":   "❓ 未知",
}


def _state_label(state: str) -> str:
    return _STATE_LABELS.get(state, f"❓ {state}" if state else "❓ 未知")


def _tab_subagents(result: ParseResult) -> None:
    if not result.subagents:
        st.info("本次会话未派发任何 subagent。")
        return

    st.subheader(f"Subagent 派发概览（共 {len(result.subagents)} 个）")

    overview_rows = []
    for sub in result.subagents:
        child_id = sub.get("childSessionID", "")
        child = _load_subagent_result(child_id)
        child_tools_df = _build_tools_df(child) if child else pd.DataFrame()
        overview_rows.append({
            "名称": sub.get("agentName") or "unnamed",
            "任务描述": sub.get("description", "") or "—",
            "状态": _state_label(sub.get("state", "")),
            "Session": (child_id[:16] + "…") if child_id else "（尚未拿到，派发中）",
            "派发 Step": sub.get("globalStep", "?"),
            "派发耗时（父侧观测）": (
                format_duration(sub["dispatchDurationMs"])
                if sub.get("dispatchDurationMs") is not None else "—"
            ),
            "峰值 Input Tokens": child.peak_input_tokens if child else None,
            "总 Output Tokens": child.result_info.total_output if child else None,
            "工具调用次数": len(child.tool_calls) if child else None,
            "成功率": f"{tool_success_rate(child_tools_df):.1f}%" if child else "—",
            "数据可用": "✅" if child and child.raw_events else "❌ 未找到 trace 文件",
        })

    df_ov = pd.DataFrame(overview_rows)
    st.dataframe(df_ov, hide_index=True, use_container_width=True)

    available = [r for r in overview_rows if r["数据可用"] == "✅"]
    if available:
        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("各 Subagent Token 消耗")
            df_tok = pd.DataFrame(available)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_tok["名称"], y=df_tok["峰值 Input Tokens"],
                                  name="Input", marker_color="#1a73e8"))
            fig.add_trace(go.Bar(x=df_tok["名称"], y=df_tok["总 Output Tokens"],
                                  name="Output", marker_color="#34a853"))
            fig.update_layout(barmode="group", height=320, margin=dict(t=10, b=0),
                               legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("各 Subagent 工具调用次数")
            df_tc = pd.DataFrame(available)
            fig2 = px.bar(df_tc, x="名称", y="工具调用次数", color="名称",
                          color_discrete_sequence=SAFE_PALETTE)
            fig2.update_layout(height=320, margin=dict(t=10, b=0), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("逐个 Subagent 详情")
    for sub in result.subagents:
        child_id = sub.get("childSessionID", "")
        name = sub.get("agentName") or "unnamed"
        session_label = f"session: {child_id[:16]}…" if child_id else "派发中，尚无子会话 ID"
        with st.expander(f"🤖 {name}  ({session_label})  {_state_label(sub.get('state', ''))}"):
            st.caption(f"派发于 Global Step {sub.get('globalStep', '?')}")
            if sub.get("description"):
                st.caption(f"任务描述：{sub['description']}")
            if sub.get("dispatchDurationMs") is not None:
                st.caption(f"父侧观测到的派发耗时：{format_duration(sub['dispatchDurationMs'])}")

            if not child_id:
                st.info("该 task 调用还没有对应的 tool.finish，子会话 ID 尚未产生，暂无法展示详情。")
                continue

            child = _load_subagent_result(child_id)
            if child is None or not child.raw_events:
                st.info("本机未找到该子会话的 trace 文件，暂无法展示 Token / 工具调用详情。")
                continue
            child_tools_df = _build_tools_df(child)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("峰值 Input Tokens", f"{child.peak_input_tokens:,}")
            c2.metric("总 Output Tokens", f"{child.result_info.total_output:,}")
            c3.metric("工具调用次数", len(child.tool_calls))
            c4.metric("工具调用成功率", f"{tool_success_rate(child_tools_df):.1f}%")
            if not child_tools_df.empty:
                tc = child_tools_df["name"].value_counts().reset_index()
                tc.columns = ["工具名称", "次数"]
                st.dataframe(tc, hide_index=True, use_container_width=True)


# ── Tab 3: Token trends ────────────────────────────────────────

def _tab_tokens(df_turns: pd.DataFrame) -> None:
    if df_turns.empty:
        st.info("暂无 Token 数据")
        return

    # Opencode stores per-step deltas; cumulate cache fields before plotting
    df_plot = df_turns.copy()
    df_plot["cache_read_cum"]     = df_plot["cache_read"].cumsum()
    df_plot["cache_creation_cum"] = df_plot["cache_creation"].cumsum()

    st.subheader("Token 消耗演进趋势")
    fig = token_trend_fig(
        df_plot,
        x_col="turn_no",
        cache_read_col="cache_read_cum",
        cache_creation_col="cache_creation_cum",
        reasoning_col="reasoning_tokens",   # cumsum'd internally
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("每轮 Token 增量（Step 差值）")
    st.plotly_chart(token_delta_fig(df_turns), use_container_width=True)


# ── Tab 4: Tool execution & consumption ─────────────────────────

def _tab_tools(df_tools: pd.DataFrame) -> None:
    if df_tools.empty:
        st.info("暂无工具执行数据")
        return

    # ── 执行情况：调用次数 / 耗时 ────────────────────────────────
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
        st.subheader("各工具耗时（avg / max）")
        dd = df_tools[df_tools["duration_ms"] > 0]
        if not dd.empty:
            ds = dd.groupby("name")["duration_ms"].agg(avg="mean", max="max").reset_index()
            fb = go.Figure()
            fb.add_trace(go.Bar(x=ds["name"], y=ds["avg"], name="平均", marker_color="#34a853"))
            fb.add_trace(go.Bar(x=ds["name"], y=ds["max"], name="最大", marker_color="#ea4335"))
            fb.update_layout(
                barmode="group", height=320, margin=dict(t=10, b=0),
                xaxis_title="工具名称", yaxis_title="ms",
                legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
            )
            st.plotly_chart(fb, use_container_width=True)
        else:
            st.info("无耗时数据")

    st.divider()
    st.subheader("每次工具调用的 Tiktoken Token 数")
    st.plotly_chart(tool_tiktoken_fig(df_tools), use_container_width=True)

    st.divider()
    st.subheader("工具效率汇总")
    tool_efficiency_table(df_tools)

    # ── 消耗排行：按物理权重分摊的 Token 消耗 ─────────────────────
    st.divider()
    st.subheader("按物理权重分摊的工具 Token 消耗排行")
    agg = (
        df_tools.groupby("name")["allotted_tokens"]
        .sum().reset_index()
        .sort_values("allotted_tokens", ascending=False)
    )
    st.plotly_chart(
        px.bar(agg, x="name", y="allotted_tokens", color="name", text_auto=True,
               labels={"name": "工具名称", "allotted_tokens": "分摊 Token 消耗"}),
        use_container_width=True,
    )

    st.divider()
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("工具单次最大 Token 消耗")
        mx = (
            df_tools.groupby("name")["allotted_tokens"]
            .max().reset_index()
            .sort_values("allotted_tokens", ascending=False)
        )
        st.plotly_chart(
            px.bar(mx, x="name", y="allotted_tokens", color="name", text_auto=True,
                   labels={"name": "工具名称", "allotted_tokens": "最大分摊 Token"}),
            use_container_width=True,
        )

    with col_d:
        st.subheader("Tiktoken Tokens vs 分摊 Token 消耗")
        fig_sc = px.scatter(
            df_tools, x="tiktoken_tokens", y="allotted_tokens",
            color="name", hover_data=["turn_no", "output_chars"],
            labels={
                "tiktoken_tokens":  "Tiktoken 估算 Tokens",
                "allotted_tokens":  "分摊 Token",
                "name":             "工具",
            },
            color_discrete_sequence=SAFE_PALETTE,
        )
        fig_sc.update_layout(height=300, margin=dict(t=10, b=0))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()
    st.subheader("逐 Step Token 分摊明细")
    step_agg = (
        df_tools.groupby("turn_no").agg(
            工具数=("name", "count"),
            总分摊Token=("allotted_tokens", "sum"),
            总TiktokenTokens=("tiktoken_tokens", "sum"),
            总输出大小chars=("output_chars", "sum"),
        ).reset_index().rename(columns={"turn_no": "Global Step"})
    )
    st.dataframe(step_agg, use_container_width=True)


# ── Tab 6: Raw data ────────────────────────────────────────────

def _tab_raw(
    result: ParseResult,
    df_turns: pd.DataFrame,
    df_tools: pd.DataFrame,
) -> None:
    df_all = pd.DataFrame(result.raw_events)
    if df_all.empty:
        st.info("无事件数据")
        return

    st.subheader(f"全部事件（{len(df_all):,} 条）")

    all_types = df_all["type"].unique().tolist() if "type" in df_all.columns else []
    type_sel = st.multiselect("事件类型", all_types, default=all_types, key="oc_raw_type")
    kw = st.text_input("关键词搜索", key="oc_raw_kw")

    df_f = df_all[df_all["type"].isin(type_sel)] if type_sel else df_all
    if kw:
        mask = df_f.apply(
            lambda row: any(kw.lower() in str(v).lower() for v in row.values), axis=1
        )
        df_f = df_f[mask]

    st.caption(f"匹配 {len(df_f):,} 条")

    # Compact columnar view (mirrors original)
    display_cols = ["type", "ts", "globalStep", "tool", "toolCallId"]
    available = [c for c in display_cols if c in df_f.columns]
    st.dataframe(
        df_f[available] if available else df_f,
        use_container_width=True,
        height=400,
    )

    # CSV export
    export = df_f.copy()
    for col in export.select_dtypes(include=["object"]).columns:
        export[col] = export[col].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
        )
    st.download_button(
        "导出 CSV",
        export.to_csv(index=False).encode("utf-8"),
        "opencode_trace.csv",
        "text/csv",
    )

    st.divider()

    if not df_turns.empty:
        st.subheader("Step 明细")
        st.dataframe(df_turns, use_container_width=True)

    if not df_tools.empty:
        st.subheader("工具调用明细")
        detail_cols = ["turn_no", "name", "duration_ms", "output_chars",
                       "tiktoken_tokens", "allotted_tokens"]
        st.dataframe(
            df_tools[[c for c in detail_cols if c in df_tools.columns]],
            use_container_width=True,
        )


# ── Sequence-diagram unit builder (bugfix) ──────────────────────

def _build_sequence_units(raw_events: list[dict]) -> list[dict]:
    """把 tool.start/tool.finish 合并为一个原子 unit 再交给 sample_events。

    原实现把 tool.start（打开 +T 激活）和 tool.finish（关闭 -T 激活）当成两条
    独立事件塞进同一个列表交给随机采样；采样经常只保留其中一条，导致 Mermaid
    收到没有闭合的激活标记而报 "Syntax error in text"。这里在采样之前把每对
    tool.start/tool.finish 合并为一个不可再分的 unit，保证激活标记总是成对
    出现或成对消失。
    """
    units: list[dict] = []
    pending: dict[str, dict] = {}
    key_types = {"text.user", "text.assistant", "step.start", "step.finish",
                 "session.start", "session.end"}
    for evt in raw_events:
        etype = evt.get("type")
        if etype == "tool.start":
            pending[evt.get("toolCallId", "")] = evt
        elif etype == "tool.finish":
            start = pending.pop(evt.get("toolCallId", ""), None)
            units.append({"kind": "tool_pair", "start": start, "finish": evt})
        elif etype in key_types:
            units.append({"kind": "single", "event": evt})
    return units


# ── Mermaid builder ────────────────────────────────────────────

def _build_mermaid(units: list[dict]) -> str:
    lines = [
        "sequenceDiagram", "    autonumber",
        "    participant U as User",
        "    participant A as Agent",
        "    participant T as Tool",
    ]
    for unit in units:
        if unit["kind"] == "tool_pair":
            start, finish = unit.get("start"), unit["finish"]
            tool_name = (start or finish).get("tool", "tool")
            lines.append(f"    A->>+T: {mermaid_quote(tool_name)}")
            err  = " [ERROR]" if finish.get("isError") else ""
            size = finish.get("outputSize", 0)
            lines.append(f"    T-->>-A: {mermaid_quote(f'done size={size}{err}')}")
            continue

        evt = unit["event"]
        etype = evt.get("type", "")
        if etype == "text.user":
            lines.append(f"    U->>+U: {mermaid_quote(evt.get('text', '')[:60])}")
            lines.append("    U-->>-U: done")
        elif etype == "text.assistant":
            lines.append(f"    A->>+A: {mermaid_quote(evt.get('text', '')[:60])}")
            lines.append("    A-->>-A: done")
        elif etype == "step.start":
            lines.append(f"    Note over A: Step {evt.get('globalStep', '?')} start")
        elif etype == "step.finish":
            reason = evt.get("reason", "")
            label  = f"Step {evt.get('globalStep', '?')} end"
            if reason:
                label += f" ({reason})"
            lines.append(f"    Note over A: {label}")
        elif etype == "session.start":
            lines.append("    Note over U,T: session start")
        elif etype == "session.end":
            lines.append("    Note over U,T: session end")
    return "\n".join(lines)
