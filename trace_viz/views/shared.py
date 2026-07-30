"""Shared UI components used across Opencode, Gemini, and Claude Code views."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trace_viz.config import MERMAID_CDN, MAX_MERMAID_EVENTS, PAGE_SIZE, SAFE_PALETTE
from trace_viz.models import ParseResult
from trace_viz.utils import mermaid_quote, sanitize_mermaid, to_str


# ── Mermaid renderer ───────────────────────────────────────────

def render_mermaid(
    mermaid_src: str,
    *,
    theme: str = "default",
    row_height: int = 32,
    event_count: int = 0,
) -> None:
    """Render a Mermaid sequence diagram in an auto-height iframe.

    We render into a temp HTML file and embed it with ``st.iframe(height="content")``.
    Streamlit then measures the iframe's real content height (the rendered svg) and
    sizes both the iframe and its wrapper to exactly that height, so the sections
    below ("复制 Mermaid 源码", "单个工具深度诊断") sit flush against the diagram -
    no overlap (diagram never taller than its box), no gap (never shorter), and the
    diagram keeps its natural aspect ratio so it stays readable.

    Previous attempts used ``components.html`` with a fixed Python height estimate:
    the estimate never matched the real svg height, so a tall diagram overflowed its
    Streamlit-owned wrapper and bled into the sections below (overlap), while a short
    one left a gap; stretching the svg with height:100% "fixed" the overlap but
    scaled the diagram down until session_end boxes were unreadable. The temp-file +
    st.iframe(height="content") path lets Streamlit own the height and keep it right.

    The HTML is written to a file (not passed inline) because st.iframe embeds
    inline HTML via srcdoc, and the minified mermaid CDN contains characters that
    break that embedding ("Invalid or unexpected token"); a real file loads normally.

    Mermaid's ``startOnLoad``/``run`` path renders into a transient detached
    container and fails inside the iframe with "svg element not in render tree",
    which Mermaid masks as the generic "Syntax error in text" error svg. We disable
    ``startOnLoad`` and call the explicit ``mermaid.render(id, src)`` API after the
    CDN has loaded, inserting the returned svg into a real, visible container.
    """
    # The source is JSON-encoded into the script so quotes, newlines, and CJK in
    # agent text cannot break out of the JS string literal - no manual escaping.
    src_json = json.dumps(mermaid_src, ensure_ascii=False)
    script = (
        "window.__mmd_src=" + src_json + ";\n"
        "mermaid.initialize({startOnLoad:false,theme:'" + theme + "',"
        "sequence:{mirrorActors:false,messageAlign:'left'}});\n"
        # Mermaid 10 leaves throwaway render artifacts in the body after each failed
        # render() retry: orphan <svg> elements (with no id) and "dmmd-<id>"
        # containers, each ~150px tall. They stack under #mermaid-out and inflate
        # the body far past the real diagram. Only remove transients that are NOT
        # inside #mermaid-out (the real diagram lives there and must survive).
        "function __mmd_cleanup(){var b=document.body,out=document.getElementById('mermaid-out');"
        "Array.prototype.forEach.call(b.children,function(n){"
        "if(n===out)return;"
        "var tn=(n.tagName||'').toUpperCase();"
        "if(tn==='SVG'||/^dmmd/.test(n.id)||(n.className&&/dmermaid/.test(n.className))){n.remove();}});}\n"
        # Mermaid 10 hard-codes actor lifeline y2=2000, so on a tall diagram (many
        # notes/messages) the vertical lines stop ~halfway and the bottom half loses
        # its column dividers. Extend every vertical lifeline (x1==x2) down to the
        # full content height after the svg lands in the out container.
        "function __mmd_extend(){var svg=document.getElementById('mermaid-out').querySelector('svg');"
        "if(!svg)return;var vb=(svg.getAttribute('viewBox')||'').split(/\\s+/);var h=vb[3]?parseFloat(vb[3]):svg.getBoundingClientRect().height;"
        "Array.prototype.forEach.call(svg.querySelectorAll('line'),function(l){"
        "var x1=l.getAttribute('x1'),x2=l.getAttribute('x2');"
        "if(x1!=null&&x2!=null&&x1===x2){l.setAttribute('y2',h);}});}\n"
        # The first mermaid.render() inside the iframe fires before the layout has
        # settled, so Mermaid's temporary svg measures against a not-yet-rendered
        # tree and throws "svg element not in render tree" (which Mermaid masks as
        # the generic "Syntax error in text" error svg). Retry with a fresh diagram
        # id until the layout settles (up to ~10s).
        "function __mmd_render(){var src=window.__mmd_src,out=document.getElementById('mermaid-out');"
        "__mmd_cleanup();"
        "var id='mmd-'+(window.__mmdTries||0);"
        "mermaid.render(id,src).then(function(r){__mmd_cleanup();out.innerHTML=r.svg;__mmd_extend();})"
        ".catch(function(e){__mmd_cleanup();"
        "window.__mmdTries=(window.__mmdTries||0)+1;"
        "if(window.__mmdTries<20){window.__mmdPending=setTimeout(__mmd_render,500);out.textContent='';}"
        "else{out.textContent='Mermaid: '+(e&&e.message||e);}});}\n"
        # The parser-blocking CDN above has loaded mermaid by here; wait for the
        # iframe to finish laying out before the first attempt.
        "window.addEventListener('load',function(){setTimeout(__mmd_render,300);});\n"
    )
    html = (
        "<!DOCTYPE html><html><head>"
        f"<script src='{MERMAID_CDN}'></script>"
        "<style>html,body{margin:0;padding:8px;background:transparent}"
        ".mermaid{display:none}"  # raw source stays hidden; the rendered svg is shown below
        ".mermaid-out{overflow:hidden}"
        # max-width:100% + height:auto keep the diagram at its natural aspect ratio
        # so it stays readable; the iframe auto-sizes to that height via height="content".
        ".mermaid-out svg{max-width:100%!important;height:auto!important;display:block}</style>"
        "</head><body>"
        "<div class='mermaid'>\n" + mermaid_src + "\n</div>"
        "<div class='mermaid-out' id='mermaid-out'></div>"
        "<script>\n" + script + "</script></body></html>"
    )
    # Write to a temp file named by a short hash of the content so that repeated
    # renders of the same diagram reuse the file (avoids writing one file per rerun)
    # while different diagrams get different files. st.iframe loads the file via a
    # real URL instead of srcdoc, which is what lets the mermaid CDN script run.
    digest = hashlib.sha1(html.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "agent_trace_vis_mermaid"
    cache_dir.mkdir(parents=True, exist_ok=True)
    html_path = cache_dir / f"mermaid-{digest}.html"
    html_path.write_text(html, encoding="utf-8")
    st.iframe(str(html_path), height="content")


def mermaid_controls(*, key_prefix: str) -> tuple[int, str, int]:
    """Render the standard mermaid control row; return (max_events, theme, row_height)."""
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        max_ev = st.slider("最大事件数", 10, 120, MAX_MERMAID_EVENTS, 5, key=f"{key_prefix}_max")
    with c2:
        theme = st.selectbox("主题", ["default", "forest", "neutral", "dark"], key=f"{key_prefix}_theme")
    with c3:
        row_h = st.slider("行高(px)", 20, 60, 30, 4, key=f"{key_prefix}_rowh")
    return max_ev, theme, row_h  # type: ignore[return-value]


def sample_events(events: list[Any], max_n: int, *, seed: int = 42) -> list[Any]:
    """Return at most max_n events, keeping first & last, sampling the middle."""
    if len(events) <= max_n:
        return events
    import random
    rng = random.Random(seed)
    middle = list(range(1, len(events) - 1))
    rng.shuffle(middle)
    chosen = sorted(middle[: max_n - 2])
    sampled = [events[0]] + [events[i] for i in chosen] + [events[-1]]
    st.info(f"共 {len(events)} 个事件，已采样显示 {len(sampled)} 个")
    return sampled


# ── Common chart builders ──────────────────────────────────────

def token_trend_fig(
    df: pd.DataFrame,
    *,
    x_col: str = "turn_no",
    input_col: str = "input_tokens",
    output_col: str = "output_tokens",
    cache_read_col: str = "cache_read",
    cache_creation_col: str = "cache_creation",
    reasoning_col: str | None = None,
) -> go.Figure:
    """Build the standard token trend figure.

    All y-columns are displayed as-is (no implicit cumsum).  Callers that
    hold per-step deltas (Opencode) should pre-cumsum those columns before
    passing them in.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[input_col],
        mode="lines+markers", name="Input（窗口大小）",
        line=dict(color="#1a73e8", width=2), marker=dict(size=8),
        fill="tozeroy", fillcolor="rgba(26,115,232,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[output_col],
        mode="lines+markers", name="Output",
        line=dict(color="#34a853", width=2), marker=dict(size=8),
    ))
    if cache_read_col in df.columns and df[cache_read_col].sum() > 0:
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[cache_read_col],
            mode="lines+markers", name="Cache Read",
            line=dict(color="#14b8a6", width=2, dash="dot"), marker=dict(size=6),
        ))
    if cache_creation_col in df.columns and df[cache_creation_col].sum() > 0:
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[cache_creation_col],
            mode="lines+markers", name="Cache Creation",
            line=dict(color="#a855f7", width=2, dash="dot"), marker=dict(size=6),
        ))
    if reasoning_col and reasoning_col in df.columns and df[reasoning_col].sum() > 0:
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[reasoning_col].cumsum(),
            mode="lines+markers", name="累计 Reasoning",
            line=dict(color="#a855f7", width=2, dash="dot"), marker=dict(size=6),
        ))
    fig.update_layout(
        height=380, hovermode="x unified",
        xaxis_title="Turn", yaxis_title="Tokens",
        margin=dict(t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def token_delta_fig(
    df: pd.DataFrame,
    *,
    x_col: str = "turn_no",
    input_col: str = "input_tokens",
    output_col: str = "output_tokens",
) -> go.Figure:
    """Per-turn input delta and output bar chart."""
    df = df.copy()
    df["_input_delta"] = df[input_col].diff().fillna(df[input_col].iloc[0]).astype(int)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[x_col], y=df["_input_delta"], name="Input 增量", marker_color="#1a73e8"))
    fig.add_trace(go.Bar(x=df[x_col], y=df[output_col],    name="Output",     marker_color="#34a853"))
    fig.update_layout(
        barmode="group", height=300,
        xaxis_title="Turn", yaxis_title="Tokens",
        margin=dict(t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def tool_tiktoken_fig(df_tools: pd.DataFrame) -> go.Figure:
    """Per-call bar chart coloured by tool name, Y = tiktoken tokens."""
    tool_color = {
        t: SAFE_PALETTE[i % len(SAFE_PALETTE)]
        for i, t in enumerate(df_tools["name"].unique())
    }
    fig = go.Figure()
    for tool_name, grp in df_tools.groupby("name"):
        fig.add_trace(go.Bar(
            x=grp.index, y=grp["tiktoken_tokens"],
            name=tool_name, marker_color=tool_color[tool_name],
            hovertemplate=f"工具: {tool_name}<br>Tiktoken: %{{y:,}}<extra></extra>",
        ))
    mean_val = df_tools["tiktoken_tokens"].mean()
    fig.add_hline(
        y=mean_val, line_dash="dot", line_color="gray",
        annotation_text=f"均值 {mean_val:.0f}", annotation_position="top right",
    )
    fig.update_layout(
        barmode="overlay", height=320, margin=dict(t=20, b=0),
        xaxis_title="第 N 次调用", yaxis_title="Tiktoken Tokens",
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
    )
    return fig


# ── Tool components ────────────────────────────────────────────

def tool_success_rate(df_tools: pd.DataFrame) -> float:
    """整体工具调用成功率（百分比）。没有调用记录时视为 100%。"""
    if df_tools.empty:
        return 100.0
    return round((1 - df_tools["is_error"].sum() / len(df_tools)) * 100, 1)


def tool_efficiency_table(df_tools: pd.DataFrame) -> None:
    """Grouped tool efficiency summary table.

    Automatically includes duration_ms stats when the column is present
    and non-zero, so both Opencode (which has duration) and Claude Code
    (which doesn't) get the right columns without extra configuration.
    """
    agg: dict[str, Any] = {
        "调用次数":         ("name",          "count"),
        "总输出chars":      ("output_chars",   "sum"),
        "均输出chars":      ("output_chars",   "mean"),
        "总TiktokenTokens": ("tiktoken_tokens","sum"),
        "均TiktokenTokens": ("tiktoken_tokens","mean"),
        "错误次数":         ("is_error",       "sum"),
    }
    has_duration = (
        "duration_ms" in df_tools.columns
        and pd.to_numeric(df_tools["duration_ms"], errors="coerce").sum() > 0
    )
    if has_duration:
        agg["平均耗时ms"] = ("duration_ms", "mean")
        agg["最大耗时ms"] = ("duration_ms", "max")

    eff = (
        df_tools.groupby("name")
        .agg(**agg)
        .reset_index()
        .rename(columns={"name": "工具"})
    )

    # 成功率：基于分组内的调用次数与错误次数计算，1-错误率
    eff["成功率"] = (
        (1 - eff["错误次数"] / eff["调用次数"].replace(0, 1)) * 100
    ).round(1)
    eff["成功率"] = eff["成功率"].apply(lambda v: f"{v:.1f}%")

    # Format numeric display columns
    for col in ("均输出chars", "均TiktokenTokens"):
        if col in eff.columns:
            eff[col] = eff[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
    for col in ("总输出chars", "总TiktokenTokens"):
        if col in eff.columns:
            eff[col] = eff[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    for col in ("平均耗时ms", "最大耗时ms"):
        if col in eff.columns:
            eff[col] = eff[col].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")

    st.dataframe(eff, hide_index=True, use_container_width=True)


def tool_inspector(df_tools: pd.DataFrame) -> None:
    """Selectbox + 2-column detail view for a single tool call."""
    if df_tools.empty:
        return
    selected = st.selectbox(
        "选择一次工具调用进行深度审查：",
        options=df_tools.to_dict(orient="records"),
        format_func=lambda x: (
            f"[Turn {x['turn_no']}] #{x['call_idx'] + 1}  {x['name']}  |  "
            f"{'❌ 错误' if x['is_error'] else '✅'}  |  "
            f"{x['tiktoken_tokens']:,} tokens"
        ),
    )
    if not selected:
        return
    c1, c2 = st.columns([1, 2])
    with c1:
        st.info(f"**Turn:** {selected['turn_no']}")
        st.info(f"**Tiktoken Tokens:** {selected['tiktoken_tokens']:,}")
        st.info(f"**输出大小:** {selected['output_chars']:,} chars")
        st.info(f"**是否错误:** {'是 ❌' if selected['is_error'] else '否 ✅'}")
        if selected.get("duration_ms"):
            st.info(f"**耗时:** {selected['duration_ms']:.0f} ms")
        raw_input = selected.get("_input_dict") or selected.get("input")
        if raw_input:
            st.subheader("入参（JSON）")
            if isinstance(raw_input, str):
                st.code(raw_input, language="json")
            else:
                st.code(json.dumps(raw_input, ensure_ascii=False, indent=2), language="json")
    with c2:
        st.subheader("工具返回内容")
        st.text_area("", value=selected.get("output", ""), height=360, label_visibility="collapsed")


# ── Generic raw-data tab ───────────────────────────────────────

def raw_events_tab(
    raw_events: list[dict],
    *,
    key_prefix: str,
    type_field: str = "type",
) -> None:
    """Paginated raw event viewer with keyword search and NDJSON export.

    Used by Claude Code view (which has no columnar DataFrame to show).
    Opencode uses its own custom raw tab for the compact dataframe layout.
    """
    all_types = sorted({str(e.get(type_field, "")) for e in raw_events})
    type_filter = st.multiselect(
        "事件类型", all_types, default=all_types, key=f"{key_prefix}_types"
    )
    kw = st.text_input("关键词搜索", key=f"{key_prefix}_kw")

    filtered = [e for e in raw_events if str(e.get(type_field, "")) in type_filter]
    if kw:
        filtered = [
            e for e in filtered
            if kw.lower() in json.dumps(e, ensure_ascii=False).lower()
        ]

    st.caption(f"匹配 {len(filtered):,} 条")

    total_pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.number_input("页码", 1, total_pages, 1, key=f"{key_prefix}_page")

    for evt in filtered[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]:
        label = str(evt.get(type_field, "?"))
        with st.expander(label):
            st.json(evt)

    st.divider()
    export = "\n".join(json.dumps(e, ensure_ascii=False) for e in filtered)
    st.download_button(
        "导出筛选结果 NDJSON",
        export.encode("utf-8"),
        f"{key_prefix}_filtered.ndjson",
        "application/x-ndjson",
    )
