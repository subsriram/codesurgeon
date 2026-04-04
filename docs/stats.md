# codesurgeon observability — query stats

codesurgeon records a row to a `query_log` SQLite table after every `run_pipeline` call.
The data is surfaced via `codesurgeon stats` (CLI) and `get_stats` (MCP tool).

---

## What is recorded

Each row captures:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | TEXT (RFC-3339) | Wall time of the call |
| `task` | TEXT | The `task` string passed by the agent |
| `intent` | TEXT | Auto-detected intent: `debug`, `refactor`, `add`, `explore`, `structural`, `general` |
| `pivot_count` | INTEGER | Number of full-source symbols returned |
| `total_tokens` | INTEGER | Tokens in the returned capsule |
| `candidate_file_tokens` | INTEGER | Token estimate for all symbols in the pivot files (baseline for savings calc) |
| `latency_ms` | INTEGER | Wall time for `build_context_capsule` (excludes auto-observation write) |
| `languages_hit` | TEXT | Comma-separated file extensions seen in pivot results, e.g. `rs,ts` |

The table is created automatically on first start via an idempotent migration in
`Database::create_schema` (`crates/cs-core/src/db.rs`).

---

## Token savings baselines

Three savings metrics are computed on-the-fly from the log:

**Candidate-file savings** (headline figure)
```
savings = (candidate_file_tokens - total_tokens) / candidate_file_tokens
```
`candidate_file_tokens` is the sum of token estimates for all symbols in the files
that produced pivots. This represents "what you would have sent if you just read the
relevant files in full."

**Workspace savings** (shown as secondary)
```
ws_savings = (workspace_tokens - avg_capsule_tokens) / workspace_tokens
```
`workspace_tokens` is the total token estimate across all indexed symbol bodies — the
absolute worst case of "send the whole codebase."

**Cost saved**
Computed from `candidate_file_tokens - total_tokens` summed over the window, priced
at claude-sonnet-4 input rates ($3 / 1M tokens). This is a rough estimate, not a
billing figure.

---

## CLI usage

```
codesurgeon stats [--days N]
```

Default look-back window: 30 days.

Sample output:
```
── Query stats (last 30 days) ──────────────────────────────────────────────
  Total queries:        47
  Token savings:        90.3%  (candidate-file baseline)
  Workspace savings:    99.1%  (avg capsule vs full workspace)
  Estimated cost saved: $0.55  (@ claude-sonnet-4 pricing)

── Latency ─────────────────────────────────────────────────────────────────
  Median: 180ms    p95: 420ms

── Intent breakdown ────────────────────────────────────────────────────────
  debug 38%  ·  add 30%  ·  refactor 21%  ·  explore 11%

── Language distribution ───────────────────────────────────────────────────
  rs 62%  ·  ts 28%  ·  sql 8%
```

---

## MCP tool

```json
{ "name": "get_stats", "arguments": { "days": 30 } }
```

Returns the same formatted report as the CLI. Useful for surfacing
"you've saved ~180k tokens this week" inline in a conversation.

---

## Implementation notes

- `run_pipeline` in `engine.rs` wraps `build_context_capsule` with `std::time::Instant`
  to measure `latency_ms`. The log write happens after the auto-observation write, so
  a failed observation does not prevent the metric from being recorded (and vice versa).
- `candidate_file_tokens` is computed by looking up all symbols for each pivot's file
  path from the DB and summing their token estimates. This is a post-hoc scan, not part
  of the search pipeline, so it does not affect latency measurements.
- The `query_log` table has no TTL — old rows accumulate indefinitely. If the database
  grows large, truncate with: `DELETE FROM query_log WHERE timestamp < datetime('now', '-90 days');`
