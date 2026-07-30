# agent_trace_vis

直接复制给Agent，让agent安装：
```bash
这个是我的dashboard项目，plugins目录下是opencode插件，可以采集opencode对话数据。现在请帮我安装这个插件，然后启动当前python项目，请仔细阅读项目的readme。
 ```

A Streamlit-based **agent trace visualizer**. It parses the internal event
streams that AI coding agents (opencode, Claude Code) produce during a session
and renders per-turn token usage, tool-call timing, context pressure, a
sequence diagram, and raw event tables — the analysis an agent author needs to
understand where tokens/time actually go.

It works in two modes:

- **Standalone** — run `streamlit run app.py` and pick a trace file via the
  sidebar (for local analysis, debugging, or the Gemini CLI telemetry format).
- **Embedded** — when iframed by [Ora](https://github.com/ora-space/desktop),
  the URL carries `?session_id=<oraSessionId>&agent_type=<…>` and the app reads
  a locator Ora wrote, resolves the trace file, parses it, and renders directly
  inside the iframe (no sidebar, no file picker). This is how the in-app
  dashboard works.

## Supported agent types

| `agent_type`  | Source of the trace data                                   | Parser                           |
| ------------- | ---------------------------------------------------------- | -------------------------------- |
| `opencode`    | `~/.local/share/opencode/trace/<session-id>.ndjson`       | `trace_viz/parsers/opencode.py`  |
| `claude_code` | `~/.claude/projects/<project-hash>/<session-id>.jsonl`     | `trace_viz/parsers/claude_code.py` |
| `gemini`      | Gemini CLI telemetry log (standalone only)                | `trace_viz/parsers/gemini.py`    |

`opencode` covers both the `opencode` and `nga` CLIs (same data model);
`claude_code` covers Claude Code's interactive transcript JSONL **and** the
`-p --output-format stream-json` one-shot format (auto-detected).

## Opencode trace-logger plugin (how the `.ndjson` is produced)

opencode does **not** write the trace-logger NDJSON natively; the file is
produced by a plugin (`plugins/trace_logger.ts`) that subscribes to the opencode
event bus and writes one NDJSON line per event.

### Install

1. Copy the plugin into the opencode plugins directory:

   ```bash
   mkdir -p ~/.config/opencode/plugins
   cp plugins/trace_logger.ts ~/.config/opencode/plugins/trace_logger.ts
   ```

2. (Optional but recommended) Register it in `~/.config/opencode/opencode.jsonc`
   if your opencode version does not auto-discover plugins in that directory:

   ```jsonc
   {
     "$schema": "https://opencode.ai/config.json",
     "plugin": ["./plugins/trace_logger.ts"]
   }
   ```

3. Start a new opencode session and use it normally. The plugin appends to:

   ```
   ~/.local/share/opencode/trace/<agent-session-id>.ndjson
   ```

   one line per event (`session.start`, `step.start`/`step.finish`,
   `tool.start`/`tool.finish`, `text.assistant`, `reasoning`, `session.end`,
   …). The file is keyed by the opencode session id (`ses_…`).

4. Visualize:
   - **Standalone** — `streamlit run app.py` → "Opencode" → upload the
     `.ndjson` file.
   - **Embedded in Ora** — open the Ora dashboard panel for that session; Ora
     resolves the file path and the app reads it automatically (no upload).

### What the plugin records

- **Tokens**: per-step deltas (`tokens{input,output,reasoning,cacheRead,
  cacheWrite}`) and cumulative totals (`cumTokens`) on every `step.finish`.
- **Tools**: `tool.start`/`tool.finish` with `args`, `output`, `outputSize`,
  `isError`, `duration`, normalized `filePath` for file tools, and separated
  `stdout`/`stderr` for `bash`.
- **Context pressure**: `contextLimit` + `contextPressure` (0–1) so the
  dashboard can show how close the session is to the context window.
- **Compaction timing**: `compaction.start`/`compaction.complete` with
  `tokensBefore`, `tokensFreed`, `compressionRatio`, `duration`.
- **Permission wait time**: `permission.asked`/`permission.replied` with
  `blockedMs` (how long the agent was blocked on the user).
- **Sub-agent spawns, retries, snapshots, file access, patches** — full event
  coverage for sequence-diagram and timeline views.

## Standalone usage

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pick a format on the landing page (Opencode / Gemini CLI / Claude Code) and
either upload a file or, for Claude Code, browse `~/.claude/projects`.

## Embedded usage (Ora integration)

When Ora's desktop app iframes this dashboard it opens:

```
http://127.0.0.1:<port>/?session_id=<oraSessionId>&agent_type=<opencode|claude_code>
```

Ora has already resolved the trace file path and written a locator to its app
data directory:

```
<appDataDir>/dashboard/<oraSessionId>.json   →   {"traceFilePath": "...", "agentType": "..."}
```

The app computes that locator root by OS convention (the Ora Tauri identifier
`space.ora.desktop`), reads the locator, reads the trace file, parses by
`agent_type`, and renders. The private agent session id never leaves Ora's
backend; only the resolved file path crosses to the dashboard.

The dashboard server itself is **externally managed** — you start Streamlit:

```bash
streamlit run app.py \
  --server.port 8601 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
```

(the `enableCORS`/`enableXsrfProtection` flags are required for the iframe to
load). Ora only knows the configured host:port and embeds it.

## Project layout

```
app.py                         # entry: embedded dispatch + standalone landing page
plugins/trace_logger.ts        # opencode plugin that produces the .ndjson
trace_viz/
  embedded.py                  # locator-root + locator + trace-file + parse dispatch (embedded)
  models.py                    # ParseResult / Turn / ToolCall / SessionInfo / ResultInfo
  parsers/
    opencode.py                # NDJSON → ParseResult (weight-distributed token allotment)
    claude_code.py             # transcript JSONL + stream-json → ParseResult
    gemini.py                  # Gemini telemetry log → ParseResult
  views/
    opencode.py                # charts: token trend, allotment, sequence, raw
    claude_code.py              # charts: timeline, token trend, cost, raw
    gemini.py                  # gemini charts
    shared.py                  # render_mermaid, token_trend_fig, tool_inspector, …
  utils.py                     # tiktoken count, mermaid sanitization, formatters
  config.py                    # palettes, page constants, mermaid CDN
sample_opencode.ndjson         # sample for offline testing
sample_claude_code.ndjson      # sample for offline testing
requirements.txt
```

## Tests

A standalone smoke test of the embedded path (locator read → trace-file read →
parse dispatch) runs against the bundled samples without a Streamlit runtime:

```bash
python -c "import sys; sys.path.insert(0,'.'); import trace_viz.embedded as e; \
  print(e.locator_root())"
```

## License

MIT.
