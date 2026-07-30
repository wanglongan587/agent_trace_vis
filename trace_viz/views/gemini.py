"""Gemini CLI telemetry visualization view."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trace_viz.config import GEM_BG, GEM_BORDER, GEM_COLORS, SAFE_PALETTE
from trace_viz.models import ParseResult
from trace_viz.parsers.gemini import parse
from trace_viz.utils import format_duration, mermaid_quote, sanitize_mermaid, to_str
from trace_viz.views.shared import (
    mermaid_controls,
    render_mermaid,
    sample_events,
)


def render() -> None:
    """Top-level entry point called from app.py."""
    result = _sidebar_input()
    if result is None:
        st.info("请在左侧选择数据来源并点击解析。")
        return

    _debug_panel(result)
    if not result.raw_events:
        st.error("解析结果为 0 条，请展开上方调试面板查看原始信息。")
        return

    elapsed = st.session_state.get("gem_elapsed", 0.0)
    st.caption(f"共 {len(result.raw_events):,} 条事件，解析耗时 {elapsed:.1f}s")

    df = _to_dataframe(result)
    _metrics_row(df)
    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["总览", "时间线", "时序图", "工具调用", "API & Tokens", "原始数据"]
    )
    with tab1: _tab_overview(df)
    with tab2: _tab_timeline(df)
    with tab3: _tab_sequence(result)
    with tab4: _tab_tools(df)
    with tab5: _tab_api_tokens(df)
    with tab6: _tab_raw(df)


# ── Sidebar: data source selection ────────────────────────────

def _sidebar_input() -> ParseResult | None:
    with st.sidebar:
        st.header("数据来源")
        if st.button("清除并重置", use_container_width=True):
            for k in [k for k in st.session_state if k.startswith("gem_")]:
                del st.session_state[k]
            st.rerun()

        mode = st.radio("方式", ["上传文件", "本地路径"])
        if mode == "上传文件":
            st.caption("默认限制 200 MB，大文件请用本地路径")
            uploaded = st.file_uploader("选择 telemetry.log", type=["log", "txt", "json"])
            if uploaded and st.button("解析", type="primary"):
                _run_parse(uploaded.read())
        else:
            path_str = st.text_input("日志路径", ".gemini/telemetry.log")
            if st.button("读取并解析", type="primary"):
                p = Path(path_str)
                if not p.exists():
                    st.error(f"文件不存在：{p}")
                else:
                    _run_parse(p.read_bytes())

        st.divider()
        st.caption("生成日志：")
        st.code(
            '$env:GEMINI_TELEMETRY_TRACES_ENABLED="true"\ngemini -p "你的任务"',
            language="powershell",
        )
    return st.session_state.get("gem_result")


def _run_parse(content: bytes) -> None:
    t0 = time.time()
    with st.spinner("解析中…"):
        result = parse(content)
    st.session_state["gem_result"]  = result
    st.session_state["gem_elapsed"] = time.time() - t0
    st.rerun()


# ── Debug panel ────────────────────────────────────────────────

def _debug_panel(result: ParseResult) -> None:
    dbg = result.parse_debug
    with st.expander("调试信息", expanded=not result.raw_events):
        st.write("**文件前 200 字符：**")
        st.code(dbg.get("first_200", ""), language="json")
        st.write("**第一个 JSON chunk 前 300 字符：**")
        st.code(dbg.get("chunk0_preview", ""), language="json")
        st.write("**chunk0 末尾 100 字符：**")
        st.code(dbg.get("chunk0_tail", ""), language="json")
        st.write(f"**chunk0 解析结果：** `{dbg.get('chunk0_parse', '')}`")
        if "chunk0_error_context" in dbg:
            st.write("**出错位置附近：**")
            st.code(dbg["chunk0_error_context"])
        st.json({k: v for k, v in dbg.items()
                 if k not in ("first_200", "chunk0_preview", "chunk0_tail",
                              "chunk0_error_context")})


# ── DataFrame builder ──────────────────────────────────────────

def _to_dataframe(result: ParseResult) -> pd.DataFrame:
    if not result.raw_events:
        return pd.DataFrame()
    df = pd.DataFrame(result.raw_events)
    for col in ("input_tokens", "output_tokens", "duration_ms", "fn_response_tokens"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp")
    return df.reset_index(drop=True)


# ── Metrics row ────────────────────────────────────────────────

def _metrics_row(df: pd.DataFrame) -> None:
    def _count(cat: str) -> int:
        return int((df.get("category", pd.Series()) == cat).sum())

    valid_ts = df["timestamp"].dropna() if "timestamp" in df.columns else pd.Series(dtype="object")
    total_dur = (
        (valid_ts.max() - valid_ts.min()).total_seconds()
        if len(valid_ts) > 1 else None
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总事件",   f"{len(df):,}")
    c2.metric("工具调用", f"{_count('工具调用'):,}")
    c3.metric("API 调用", f"{_count('API 调用'):,}")
    c4.metric("文件操作", f"{_count('文件操作'):,}")
    c5.metric("总耗时",   f"{total_dur:.1f}s" if total_dur else "—")


# ── Tab 1: Overview ────────────────────────────────────────────

def _tab_overview(df: pd.DataFrame) -> None:
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("事件类型分布")
        if "category" in df.columns:
            cc = df["category"].value_counts().reset_index()
            cc.columns = ["category", "count"]
            fig = px.pie(cc, names="category", values="count",
                         color="category", color_discrete_map=GEM_COLORS, hole=0.4)
            fig.update_traces(textinfo="label+percent+value")
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("事件名称频次 Top 15")
        if "event_name" in df.columns:
            nc = df["event_name"].value_counts().head(15).reset_index()
            nc.columns = ["event_name", "count"]
            nc["short"] = nc["event_name"].str.replace(
                r"^(gemini_cli\.|gen_ai\.client\.)", "", regex=True)
            fig2 = px.bar(nc, x="count", y="short", orientation="h",
                          color_discrete_sequence=["#1a73e8"])
            fig2.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=0, b=0))
            st.plotly_chart(fig2, use_container_width=True)

    df_ts = df[df["timestamp"].notna()] if "timestamp" in df.columns else pd.DataFrame()
    if not df_ts.empty:
        st.subheader("事件时间分布（按分钟聚合）")
        df_ts = df_ts.copy()
        df_ts["minute"] = df_ts["timestamp"].dt.floor("min")
        mc = df_ts.groupby(["minute", "category"]).size().reset_index(name="count")
        fig3 = px.bar(mc, x="minute", y="count", color="category",
                      color_discrete_map=GEM_COLORS, barmode="stack")
        fig3.update_layout(margin=dict(t=0, b=0), height=260)
        st.plotly_chart(fig3, use_container_width=True)

    if "model" in df.columns:
        models = df[df["model"].notna() & (df["model"] != "")]["model"].unique()
        if len(models):
            st.info(f"使用模型：{', '.join(models)}")

    with st.expander("所有事件名及其分类（用于核对计数）"):
        if {"event_name", "category"}.issubset(df.columns):
            name_cat = (df.groupby(["event_name", "category"])
                          .size().reset_index(name="count")
                          .sort_values("count", ascending=False))
            st.dataframe(name_cat, hide_index=True, use_container_width=True, height=300)


# ── Tab 2: Timeline ────────────────────────────────────────────

def _tab_timeline(df: pd.DataFrame) -> None:
    st.subheader("事件时间线")
    cats = df["category"].unique().tolist() if "category" in df.columns else []
    cats_sel = st.multiselect("类型", cats, default=cats, key="gem_tl_c")
    kw = st.text_input("关键词", key="gem_tl_kw", placeholder="事件名 / body / 属性值")

    df_f = df[df["category"].isin(cats_sel)] if cats_sel else df
    if kw:
        mask = (
            df_f.get("event_name", pd.Series()).str.contains(kw, case=False, na=False)
            | df_f.get("body",       pd.Series()).str.contains(kw, case=False, na=False)
            | df_f.get("attrs_json", pd.Series()).str.contains(kw, case=False, na=False)
        )
        df_f = df_f[mask]
    st.caption(f"匹配 {len(df_f):,} 条")

    df_p = df_f[df_f["timestamp"].notna()] if "timestamp" in df_f.columns else pd.DataFrame()
    if not df_p.empty:
        PLOT_LIMIT = 5000
        sample = df_p.sample(PLOT_LIMIT, random_state=42) if len(df_p) > PLOT_LIMIT else df_p
        if len(df_p) > PLOT_LIMIT:
            st.info(f"超过 {PLOT_LIMIT} 条，散点图随机采样")
        fig_tl = px.scatter(sample, x="timestamp", y="category",
                            color="category", color_discrete_map=GEM_COLORS,
                            hover_data={"event_name": True, "body": True, "timestamp": False})
        fig_tl.update_traces(marker=dict(size=8, opacity=0.7))
        fig_tl.update_layout(showlegend=False, height=260, margin=dict(t=10, b=10))
        st.plotly_chart(fig_tl, use_container_width=True)

    from trace_viz.config import PAGE_SIZE
    total_p = max(1, (len(df_f) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.number_input("页码", 1, total_p, 1, key="gem_tl_page")
    st.caption(f"第 {page}/{total_p} 页")

    for _, row in df_f.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE].iterrows():
        ts_s = (row["timestamp"].strftime("%H:%M:%S.%f")[:-3]
                if "timestamp" in row.index and pd.notna(row.get("timestamp")) else "")
        short = re.sub(r"^(gemini_cli\.|gen_ai\.client\.)", "",
                       str(row.get("event_name", "")))
        cat    = str(row.get("category", "其他"))
        bg     = GEM_BG.get(cat,     GEM_BG.get("其他", ""))
        border = GEM_BORDER.get(cat, GEM_BORDER.get("其他", "#888"))

        with st.expander(f"[{cat}]  {short}    {ts_s}"):
            if row.get("body"):
                st.markdown(
                    f'<div style="background:{bg};border-left:3px solid {border};'
                    f'border-radius:6px;padding:6px 10px;margin-bottom:8px;'
                    f'font-size:13px;color:#444;">{row["body"]}</div>',
                    unsafe_allow_html=True,
                )
            try:
                attrs = json.loads(str(row.get("attrs_json", "{}")))
                if attrs:
                    rows_html = "".join(
                        f"<tr><td style='color:#666;font-family:monospace;padding:3px 10px 3px 0;"
                        f"white-space:nowrap;vertical-align:top'>{k}</td>"
                        f"<td style='padding:3px 0;word-break:break-all'>{v}</td></tr>"
                        for k, v in attrs.items()
                    )
                    st.markdown(
                        f'<div style="background:{bg};border-left:3px solid {border};'
                        f'border-radius:6px;padding:8px 12px;">'
                        f'<table style="width:100%;font-size:12px;border-collapse:collapse">'
                        f'{rows_html}</table></div>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                st.code(str(row.get("attrs_json", "")))


# ── Tab 3: Sequence diagram ────────────────────────────────────

def _tab_sequence(result: ParseResult) -> None:
    st.subheader("主要步骤时序图")
    max_ev, theme, row_h = mermaid_controls(key_prefix="gem_seq")

    df_seq = _to_dataframe(result)
    if "timestamp" in df_seq.columns:
        valid = df_seq["timestamp"].dropna()
        if len(valid) > 1:
            t_min = valid.min().to_pydatetime()
            t_max = valid.max().to_pydatetime()
            if t_min < t_max:
                sel = st.slider("时间范围", min_value=t_min, max_value=t_max,
                                value=(t_min, t_max), format="HH:mm:ss",
                                key="gem_seq_time")
                df_seq = df_seq[
                    (df_seq["timestamp"] >= sel[0]) & (df_seq["timestamp"] <= sel[1])
                ]

    steps   = _extract_sequence_steps(df_seq)
    sampled = sample_events(steps, max_ev)
    if not sampled:
        st.warning("未找到可渲染的关键事件")
        return

    src = _build_mermaid(sampled)
    render_mermaid(src, theme=theme, row_height=row_h, event_count=len(sampled))
    with st.expander("复制 Mermaid 源码（可粘贴到 Notion / GitHub）"):
        st.code(src, language="text")


# ── Tab 4: Tools ───────────────────────────────────────────────

def _tab_tools(df: pd.DataFrame) -> None:
    if "category" not in df.columns:
        st.info("无分类数据")
        return

    df_tool  = df[df["category"] == "工具调用"].copy()
    df_tresp = df[df["category"] == "工具响应"].copy()

    if df_tool.empty:
        st.info("未找到工具调用事件")
        return

    for col in ("duration_ms", "fn_response_tokens"):
        if col in df_tool.columns:
            df_tool[col] = pd.to_numeric(df_tool[col], errors="coerce")
    if not df_tresp.empty and "fn_response_tokens" in df_tresp.columns:
        df_tresp["fn_response_tokens"] = pd.to_numeric(
            df_tresp["fn_response_tokens"], errors="coerce"
        )

    # ── Pair response tokens ───────────────────────────────────
    has_tok_in_call = (
        "fn_response_tokens" in df_tool.columns
        and df_tool["fn_response_tokens"].notna().any()
    )
    has_tok_in_resp = (
        not df_tresp.empty
        and "fn_response_tokens" in df_tresp.columns
        and df_tresp["fn_response_tokens"].notna().any()
    )
    has_resp_tokens = has_tok_in_call or has_tok_in_resp

    if has_tok_in_call:
        df_calls = df_tool.copy()
        df_calls["resp_tokens"] = df_calls["fn_response_tokens"]
    elif has_tok_in_resp:
        tc = df_tool.sort_values("timestamp").reset_index(drop=True).copy() \
            if "timestamp" in df_tool.columns else df_tool.reset_index(drop=True).copy()
        tr = df_tresp.sort_values("timestamp").reset_index(drop=True).copy() \
            if "timestamp" in df_tresp.columns else df_tresp.reset_index(drop=True).copy()
        n  = min(len(tc), len(tr))
        tc = tc.iloc[:n].copy()
        tc["resp_tokens"] = tr["fn_response_tokens"].values[:n]
        df_calls = tc
    else:
        df_calls = df_tool.copy()
        df_calls["resp_tokens"] = float("nan")

    df_calls["call_no"] = range(1, len(df_calls) + 1)
    fname = "function_name" if "function_name" in df_calls.columns else "tool_name"

    # ── Debug expander when response tokens cannot be found ────
    if not has_resp_tokens:
        with st.expander("未检测到 response token 字段，点击查看原始属性"):
            for src_df, label in [(df_tool, "工具调用"), (df_tresp, "工具响应")]:
                if src_df.empty or "_token_attrs" not in src_df.columns:
                    continue
                sample = src_df[src_df["_token_attrs"].notna()].head(2)
                for _, r in sample.iterrows():
                    st.caption(f"[{label}] 工具：{r.get(fname, '?')}")
                    try:
                        st.json(json.loads(r["_token_attrs"]))
                    except Exception:
                        st.code(str(r["_token_attrs"]))

    # ── Metrics ────────────────────────────────────────────────
    avg_dur    = df_tool["duration_ms"].mean()    if "duration_ms"    in df_tool.columns else float("nan")
    max_dur    = df_tool["duration_ms"].max()     if "duration_ms"    in df_tool.columns else float("nan")
    total_resp = df_calls["resp_tokens"].sum(skipna=True) if has_resp_tokens else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("总调用次数",         f"{len(df_tool):,}")
    m2.metric("平均耗时",           f"{avg_dur:.0f} ms"  if pd.notna(avg_dur)  else "—")
    m3.metric("最长耗时",           f"{max_dur:.0f} ms"  if pd.notna(max_dur)  else "—")
    m4.metric("Response Tokens 合计",
              f"{int(total_resp):,}" if total_resp and pd.notna(total_resp) else "—")

    st.divider()

    # ── Call count + duration charts ───────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("各工具调用次数")
        tc = df_calls[fname].value_counts().reset_index()
        tc.columns = ["工具", "次数"]
        fa = px.bar(tc, x="工具", y="次数", color="工具", text="次数",
                    color_discrete_sequence=SAFE_PALETTE)
        fa.update_traces(textposition="outside")
        fa.update_layout(height=320, margin=dict(t=10, b=0), showlegend=False)
        st.plotly_chart(fa, use_container_width=True)

    with col_b:
        st.subheader("各工具耗时（avg / P90 / max）")
        dd = df_calls[df_calls["duration_ms"].notna()] if "duration_ms" in df_calls.columns else pd.DataFrame()
        if not dd.empty:
            ds = dd.groupby(fname)["duration_ms"].agg(
                avg="mean",
                p90=lambda x: x.quantile(0.9),
                max="max",
            ).reset_index()
            fb = go.Figure()
            fb.add_trace(go.Bar(x=ds[fname], y=ds["avg"], name="平均", marker_color="#34a853"))
            fb.add_trace(go.Bar(x=ds[fname], y=ds["p90"], name="P90",  marker_color="#fbbc04"))
            fb.add_trace(go.Bar(x=ds[fname], y=ds["max"], name="最大", marker_color="#ea4335"))
            fb.update_layout(barmode="group", height=320, margin=dict(t=10, b=0),
                             legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
            st.plotly_chart(fb, use_container_width=True)
        else:
            st.info("无耗时数据")

    st.divider()

    # ── Per-call response token bar + summary charts ───────────
    st.subheader("每次工具调用的 Response Token 数量")
    if has_resp_tokens and df_calls["resp_tokens"].notna().any():
        rt = df_calls[df_calls["resp_tokens"].notna()].copy()
        tool_color = {
            t: SAFE_PALETTE[i % len(SAFE_PALETTE)]
            for i, t in enumerate(rt[fname].unique())
        }

        # Per-call bar
        fc2 = go.Figure()
        for tn, grp in rt.groupby(fname):
            fc2.add_trace(go.Bar(
                x=grp["call_no"], y=grp["resp_tokens"],
                name=tn, marker_color=tool_color[tn],
                hovertemplate=f"工具: {tn}<br>Tokens: %{{y:,}}<extra></extra>",
            ))
        mean_tok = rt["resp_tokens"].mean()
        fc2.add_hline(y=mean_tok, line_dash="dot", line_color="gray",
                      annotation_text=f"均值 {mean_tok:.0f}",
                      annotation_position="top right")
        fc2.update_layout(
            barmode="overlay", height=340, margin=dict(t=20, b=0),
            xaxis_title="第 N 次工具调用", yaxis_title="Response Tokens",
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        )
        st.plotly_chart(fc2, use_container_width=True)

        # col_c: per-tool summary  |  col_d: trend scatter + rolling mean
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("各工具 Response Token 汇总")
            rt_stat = (
                rt.groupby(fname)["resp_tokens"]
                .agg(总计="sum", 均值="mean", 最大="max")
                .reset_index()
                .rename(columns={fname: "工具"})
            )
            fd = go.Figure()
            fd.add_trace(go.Bar(
                x=rt_stat["工具"], y=rt_stat["总计"],
                name="总计", marker_color="#1a73e8",
                text=rt_stat["总计"].apply(lambda v: f"{v:.0f}"),
                textposition="outside",
            ))
            fd.add_trace(go.Bar(
                x=rt_stat["工具"], y=rt_stat["均值"],
                name="均值/次", marker_color="#a8c7fa",
                text=rt_stat["均值"].apply(lambda v: f"{v:.0f}"),
                textposition="outside",
            ))
            fd.update_layout(
                barmode="group", height=300, margin=dict(t=10, b=0),
                xaxis_title="工具名称", yaxis_title="Tokens",
                legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
            )
            st.plotly_chart(fd, use_container_width=True)

        with col_d:
            st.subheader("逐次 Response Token 趋势")
            fp = px.scatter(
                rt, x="call_no", y="resp_tokens",
                color=fname,
                hover_data={"file_path": True, "call_no": False} if "file_path" in rt.columns else {},
                labels={
                    "call_no":     "第 N 次调用",
                    "resp_tokens": "Response Tokens",
                    fname:         "工具",
                },
            )
            fp.add_trace(go.Scatter(
                x=rt["call_no"],
                y=rt["resp_tokens"].rolling(5, min_periods=1).mean(),
                mode="lines", name="滚动均值(5)",
                line=dict(color="gray", width=1.5, dash="dot"),
            ))
            fp.update_layout(height=300, margin=dict(t=10, b=0))
            st.plotly_chart(fp, use_container_width=True)

    else:
        st.info(
            "未找到 Response Token 数据。"
            "展开上方调试面板查看原始字段名，发给我即可更新解析逻辑。"
        )

    st.divider()

    # ── Efficiency table + slowest calls ───────────────────────
    col_e, col_f = st.columns(2)

    with col_e:
        st.subheader("工具效率汇总")
        agg_spec: dict = {"调用次数": (fname, "count")}
        if "duration_ms" in df_calls.columns:
            agg_spec["平均耗时ms"]  = ("duration_ms", "mean")
            agg_spec["最大耗时ms"]  = ("duration_ms", "max")
        if "resp_tokens" in df_calls.columns:
            agg_spec["总ResponseTokens"] = ("resp_tokens", "sum")
            agg_spec["均ResponseTokens"] = ("resp_tokens", "mean")
        eff = (
            df_calls.groupby(fname).agg(**agg_spec)
            .reset_index()
            .rename(columns={fname: "工具"})
        )
        for col in ("平均耗时ms", "最大耗时ms", "均ResponseTokens"):
            if col in eff.columns:
                eff[col] = eff[col].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
        st.dataframe(eff, hide_index=True, use_container_width=True)

    with col_f:
        st.subheader("最慢的 10 次调用")
        if "duration_ms" in df_calls.columns:
            slow_cols = [fname, "duration_ms"]
            if "file_path"   in df_calls.columns: slow_cols.insert(1, "file_path")
            if "timestamp"   in df_calls.columns: slow_cols.append("timestamp")
            if "resp_tokens" in df_calls.columns: slow_cols.append("resp_tokens")
            slow = (
                df_calls[df_calls["duration_ms"].notna()]
                .nlargest(10, "duration_ms")[slow_cols]
                .reset_index(drop=True)
            )
            slow.rename(columns={fname: "工具", "duration_ms": "耗时(ms)"}, inplace=True)
            slow["耗时(ms)"] = slow["耗时(ms)"].apply(lambda x: f"{x:.0f}")
            st.dataframe(slow, hide_index=True, use_container_width=True)
        else:
            st.info("无耗时数据")


# ── Tab 5: API & Tokens ────────────────────────────────────────

def _tab_api_tokens(df: pd.DataFrame) -> None:
    df_api = df[df.get("category", pd.Series()) == "API 调用"].copy() \
        if "category" in df.columns else pd.DataFrame()
    if df_api.empty:
        st.info("未找到 API 调用事件")
        return

    df_api["input_tokens"]  = pd.to_numeric(df_api.get("input_tokens"),  errors="coerce").fillna(0)
    df_api["output_tokens"] = pd.to_numeric(df_api.get("output_tokens"), errors="coerce").fillna(0)
    ti  = int(df_api["input_tokens"].sum())
    to_ = int(df_api["output_tokens"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("API 调用次数",  f"{len(df_api):,}")
    c2.metric("总 Input Tokens",  f"{ti:,}")
    c3.metric("总 Output Tokens", f"{to_:,}")
    c4.metric("合计",             f"{ti + to_:,}")

    if ti + to_ == 0:
        return

    ds = (
        df_api[df_api["timestamp"].notna()].sort_values("timestamp").reset_index(drop=True)
        if "timestamp" in df_api.columns else df_api.reset_index(drop=True)
    )
    ds_tok = ds[(ds["input_tokens"] > 0) | (ds["output_tokens"] > 0)].copy()
    ds_tok["call_no"]  = range(1, len(ds_tok) + 1)
    ds_tok["累计输入"] = ds_tok["input_tokens"].cumsum()
    ds_tok["累计输出"] = ds_tok["output_tokens"].cumsum()

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("输入 / 输出总量")
        fb = go.Figure(go.Bar(
            x=["输入", "输出"], y=[ti, to_],
            marker_color=["#1a73e8", "#34a853"],
            text=[f"{ti:,}", f"{to_:,}"], textposition="outside",
        ))
        fb.update_layout(height=320, margin=dict(t=20, b=0),
                         showlegend=False, yaxis_title="Tokens")
        st.plotly_chart(fb, use_container_width=True)

    with col_b:
        st.subheader("每次调用的 Input Token 数")
        fi = go.Figure(go.Bar(
            x=ds_tok["call_no"], y=ds_tok["input_tokens"], marker_color="#1a73e8",
        ))
        fi.update_layout(height=320, margin=dict(t=20, b=0),
                         xaxis_title="第 N 次 API 调用",
                         yaxis_title="Input Tokens", showlegend=False)
        st.plotly_chart(fi, use_container_width=True)

    st.divider()
    st.subheader("累计 Token 消耗曲线（按时间）")
    x_col = "timestamp" if "timestamp" in ds_tok.columns else "call_no"
    fc = go.Figure()
    fc.add_trace(go.Scatter(x=ds_tok[x_col], y=ds_tok["累计输入"],
                            mode="lines+markers", name="累计输入",
                            line=dict(color="#1a73e8", width=2),
                            fill="tozeroy", fillcolor="rgba(26,115,232,0.08)"))
    fc.add_trace(go.Scatter(x=ds_tok[x_col], y=ds_tok["累计输出"],
                            mode="lines+markers", name="累计输出",
                            line=dict(color="#34a853", width=2),
                            fill="tozeroy", fillcolor="rgba(52,168,83,0.08)"))
    fc.update_layout(height=320, margin=dict(t=10, b=0), yaxis_title="Tokens",
                     hovermode="x unified",
                     legend=dict(orientation="h", yanchor="bottom",
                                 y=1.02, xanchor="right", x=1))
    st.plotly_chart(fc, use_container_width=True)

    st.divider()
    st.subheader("每次调用 Input vs Output 对比")
    fg = go.Figure()
    fg.add_trace(go.Bar(x=ds_tok["call_no"], y=ds_tok["input_tokens"],
                        name="Input", marker_color="#1a73e8"))
    fg.add_trace(go.Bar(x=ds_tok["call_no"], y=ds_tok["output_tokens"],
                        name="Output", marker_color="#34a853"))
    fg.update_layout(barmode="group", height=320, margin=dict(t=10, b=0),
                     xaxis_title="第 N 次 API 调用", yaxis_title="Tokens",
                     legend=dict(orientation="h", yanchor="bottom",
                                 y=1.02, xanchor="right", x=1))
    st.plotly_chart(fg, use_container_width=True)


# ── Tab 6: Raw data ────────────────────────────────────────────

def _tab_raw(df: pd.DataFrame) -> None:
    st.subheader(f"全部事件（{len(df):,} 条）")
    cats = df["category"].unique().tolist() if "category" in df.columns else []
    cat2 = st.multiselect("类型", cats, default=cats, key="gem_raw_c")
    kw2  = st.text_input("搜索", key="gem_raw_kw")

    dr = df[df["category"].isin(cat2)] if cat2 else df
    if kw2:
        mask = (
            dr.get("event_name", pd.Series()).str.contains(kw2, case=False, na=False)
            | dr.get("body",       pd.Series()).str.contains(kw2, case=False, na=False)
            | dr.get("attrs_json", pd.Series()).str.contains(kw2, case=False, na=False)
        )
        dr = dr[mask]

    display_cols = [c for c in
                    ["timestamp", "category", "event_name", "body", "tool_name",
                     "file_path", "input_tokens", "output_tokens", "duration_ms", "status"]
                    if c in dr.columns]
    st.dataframe(dr[display_cols].reset_index(drop=True),
                 use_container_width=True, height=500)
    st.download_button(
        "导出 CSV",
        dr[display_cols].to_csv(index=False).encode("utf-8"),
        "gemini_telemetry.csv",
        "text/csv",
    )


# ── Mermaid sequence ───────────────────────────────────────────

def _extract_sequence_steps(df: pd.DataFrame) -> list[dict]:
    def etype(name: str) -> str | None:
        if "prompt"          in name: return "prompt"
        if "agent_run_start" in name: return "agent_start"
        if "agent_run_end"   in name: return "agent_end"
        if "tool_call"       in name: return "tool_call"
        if "file_operation"  in name: return "file_op"
        if name.startswith("gen_ai"): return "api_call"
        return None

    steps = []
    for _, row in df.iterrows():
        et = etype(str(row.get("event_name", "")))
        if not et:
            continue
        try:
            attrs = json.loads(str(row.get("attrs_json", "{}")))
        except Exception:
            attrs = {}
        ts = (row["timestamp"].strftime("%H:%M:%S")
              if "timestamp" in row.index and pd.notna(row.get("timestamp")) else "")
        steps.append({"etype": et, "ts": ts, "attrs": attrs, "row": row})
    return steps


def _build_mermaid(steps: list[dict]) -> str:
    lines = [
        "sequenceDiagram", "    autonumber",
        "    participant U as User",
        "    participant A as Agent",
        "    participant L as LLM API",
        "    participant T as Tool",
        "    participant F as FileSystem",
    ]

    def _short_path(p: str, n: int = 32) -> str:
        p = sanitize_mermaid(p, max_len=255)
        return ("…" + p[-n:]) if len(p) > n else p

    for step in steps:
        et    = step["etype"]
        attrs = step["attrs"]
        row   = step["row"]

        if et == "prompt":
            pl = attrs.get("prompt_length", "")
            lines.append(
                f"    U->>A: {mermaid_quote(f'提交任务 长度={pl}chars' if pl else '提交任务')}"
            )
        elif et == "agent_start":
            turn = attrs.get("turn", attrs.get("turn_number", ""))
            lines.append(
                f"    Note over A: {mermaid_quote(f'Agent开始 turn={turn}' if turn else 'Agent开始运行')}"
            )
        elif et == "api_call":
            in_t   = attrs.get("input_tokens",  attrs.get("gen_ai.usage.input_tokens",  ""))
            out_t  = attrs.get("output_tokens", attrs.get("gen_ai.usage.output_tokens", ""))
            finish = sanitize_mermaid(
                to_str(attrs.get("finish_reason",
                                 attrs.get("gen_ai.response.finish_reasons", ""))), 15
            )
            parts = []
            if in_t:   parts.append(f"in={in_t}tok")
            if out_t:  parts.append(f"out={out_t}tok")
            if finish: parts.append(f"finish={finish}")
            resp = "返回结果" + (" " + " ".join(parts) if parts else "")
            lines.append(f"    A->>+L: {mermaid_quote('LLM推理')}")
            lines.append(f"    L-->>-A: {mermaid_quote(resp)}")
        elif et == "tool_call":
            fname   = "function_name" if "function_name" in row.index else "tool_name"
            fn      = sanitize_mermaid(
                attrs.get("function_name", to_str(row.get(fname, "tool"))), 22
            )
            dur     = attrs.get("duration_ms", "")
            success = to_str(attrs.get("success", attrs.get("status", "")))
            fpath   = _short_path(to_str(attrs.get("file_path", row.get("file_path", ""))))
            req     = fn + (f" path={fpath}" if fpath else "")
            resp_parts = []
            if dur: resp_parts.append(f"{dur}ms")
            if success in ("True", "true", "success"):
                resp_parts.append("成功")
            elif success and success not in ("", "None", "False"):
                resp_parts.append(sanitize_mermaid(success, 15))
            lines.append(f"    A->>+T: {mermaid_quote(req)}")
            lines.append(
                f"    T-->>-A: {mermaid_quote('完成 ' + ' '.join(resp_parts) if resp_parts else '完成')}"
            )
        elif et == "file_op":
            op    = attrs.get("operation", "").lower()
            fpath = _short_path(to_str(attrs.get("path", attrs.get("file_path", ""))))
            op_zh = {"read": "读取", "write": "写入", "delete": "删除"}.get(op, op or "操作")
            size  = attrs.get("size_bytes", "")
            size_str = ""
            if size:
                try:
                    kb = int(float(size)) // 1024
                    size_str = f" {kb}KB" if kb > 0 else f" {int(float(size))}B"
                except Exception:
                    pass
            lines.append(f"    T->>F: {mermaid_quote(f'{op_zh} {fpath}{size_str}' if fpath else op_zh)}")
        elif et == "agent_end":
            status = to_str(attrs.get("status", ""))
            turns  = attrs.get("total_turns", attrs.get("turn_count", ""))
            dur    = attrs.get("duration_ms", "")
            parts  = ["任务结束"]
            if status: parts.append(f"状态={sanitize_mermaid(status, 12)}")
            if turns:  parts.append(f"共{turns}轮")
            if dur:
                try: parts.append(f"耗时{int(float(dur)) // 1000}s")
                except Exception: pass
            lines.append(f"    A->>U: {mermaid_quote(' '.join(parts))}")

    return "\n".join(lines)
