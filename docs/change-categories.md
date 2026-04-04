# codesurgeon AST change categories

> **Keep this doc up to date whenever diffing logic in `reindex_file` changes.**
> The pipeline lives in `crates/cs-core/src/engine.rs::reindex_file` and
> `crates/cs-core/src/memory.rs::record_file_edit`.

---

## Overview

When a file is re-indexed after a change, codesurgeon diffs the old symbol list
(snapshotted before deletion) against the newly parsed symbols. Each detected
change is classified into one of five categories and stored in the `change_category`
column of the resulting `passive` observation.

This lets agents answer "what kind of change happened to this file?" rather than just
"this file was touched."

---

## Categories

| Category | Condition | Notes |
|---|---|---|
| `new_symbol` | FQN present in new parse, absent in old | Non-import symbols (functions, classes, structs, …) |
| `deleted_symbol` | FQN present in old parse, absent in new | Any symbol kind |
| `signature_change` | Same FQN, different `signature` field | Covers parameter changes, return-type annotations visible in the signature |
| `body_change` | Same FQN, same `signature`, different `body` | Implementation changed; signature stable |
| `dependency_added` | New FQN with `SymbolKind::Import` | `import`, `use`, `require`, etc. |

### Priority order (for the single per-observation tag)

When a file edit touches symbols across multiple categories, one observation is
written with a human-readable content breakdown and the **highest-priority**
category set as `change_category`:

```
new_symbol > deleted_symbol > signature_change > dependency_added > body_change
```

Content example:
```
File changed: src/engine.rs — signature_change: CoreEngine::reindex_file; body_change: CoreEngine::new, … (4 total)
```

---

## Deferred / not implemented

- **`type_change`** — distinguishing a return-type-only change from a broader
  signature change requires language-specific parsing. Deferred; these currently
  appear as `signature_change`.
- **`renamed`** — fragile to detect without LSP resolution (line-number shifts on
  reformats). Not implemented.
- **Visibility changes** — language-specific; deferred.

---

## Surfacing in tools

| Tool | Behaviour |
|---|---|
| `get_session_context` | `change_category` shown as `[cat]` badge on each passive observation |
| `search_memory` | `change_category` is searchable via keyword (it appears in observation content) |
| `get_diff_capsule` | Passive observations are included; `change_category` badge visible at all detail levels |

---

## Implementation

| Component | Location |
|---|---|
| Diff logic | `engine.rs::reindex_file` — phases 1 and 3 |
| `SymbolChange` type | `memory.rs::SymbolChange` |
| `record_file_edit` | `memory.rs::MemoryStore::record_file_edit` |
| DB column | `observations.change_category TEXT` (nullable, idempotent migration) |
| Display | `cs-mcp/src/main.rs::format_observations` |
