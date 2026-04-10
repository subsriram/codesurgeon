# codesurgeon indexing lifecycle

> **Keep this doc up to date whenever indexing, file-watching, or pruning logic changes.**
> The pipeline lives in `crates/cs-core/src/engine.rs::index_workspace_inner` and
> `crates/cs-core/src/engine.rs::reindex_file`.

---

## Overview

codesurgeon maintains a persistent index of every symbol in the workspace across six stores.
Two entry points keep the index current:

| Entry point | Trigger | Scope |
|-------------|---------|-------|
| `index_workspace()` | Server startup | Full workspace scan + stale file pruning |
| `reindex_file()` | File watcher event | Single file (create / modify / delete) |

Both paths must keep all six stores in sync:

```
Stores
  ├── SQLite symbols table    persistent symbol metadata
  ├── SQLite symbols_fts      FTS5 full-text search
  ├── SQLite edges table      static call / import / type edges
  ├── SQLite symbol_embeddings  vector embeddings (embeddings build only)
  ├── Tantivy (in-memory)     BM25 search index
  └── petgraph (in-memory)    dependency graph + centrality caches
```

---

## Full index (`index_workspace`)

Runs once at server startup. Steps:

```
1. Collect source files on disk
2. Load baseline file hashes from SQLite (incremental skip)
3. Prune stale files  ← files tracked in DB but missing from disk
4. Parse changed files in parallel (rayon)
5. Flush to SQLite (single transaction)
6. Rebuild in-memory graph + Tantivy
7. Enrichment passes (macros, rustdoc, TypeScript, LSP edges)
8. Embed symbols (embeddings build only)
9. Write manifest
```

### Stale file pruning (step 3)

Before parsing, `index_workspace` diffs the `files` table against the on-disk file set.
Any file tracked in the DB but absent from disk is fully purged:

- `edges` rows referencing its symbol IDs
- `symbol_embeddings` rows for its symbol IDs
- `symbols` + `symbols_fts` rows
- `lsp_edges` rows from that file
- `files` table entry
- Tantivy terms for each symbol ID
- petgraph nodes (implicit edge removal)

This catches bulk deletions that the file watcher may miss: deleted git worktrees,
`git checkout` switching branches, manual `rm -rf`, or files removed while the server
was not running.

---

## Incremental reindex (`reindex_file`)

Called by the file watcher (`notify` crate) on every create, modify, or delete event.

### Modify / Create path

```
Phase 1  [db lock]     Snapshot old symbols, delete old edges + embeddings + symbols + LSP edges
Phase 2  [graph lock]  Remove old file from in-memory graph
Phase 3  [no locks]    Parse file into new symbol list, compute content hash
Phase 4  [db lock]     Write new symbols + file hash in a single transaction
Phase 5  [graph+search lock]  Add new symbols to graph + Tantivy, commit
Phase 6  [memory lock] Record file-edit observation for session memory
Phase 7  [db lock]     Re-embed new symbols, refresh embedding cache (embeddings build only)
```

### Delete path (`ChangeKind::Removed`)

Phases 1–2 run the same cleanup as modify. Then:

```
Phase 2b [search lock]  Delete symbol terms from Tantivy, commit
         [db lock]      Delete file entry from files table
         [emb]          Refresh embedding cache (embeddings build only)
         return early
```

All six stores are cleaned up. No orphaned rows or ghost search results remain.

---

## What each store cleanup does

| Store | Cleanup method | Notes |
|-------|---------------|-------|
| `symbols` + `symbols_fts` | `db.delete_file_symbols()` | FTS rows deleted first by rowid |
| `edges` | `db.delete_edges_for_symbols()` | Deletes both `from_id` and `to_id` references |
| `symbol_embeddings` | `db.delete_embeddings_for_symbols()` | By symbol ID |
| `lsp_edges` | `db.delete_lsp_edges_for_file()` | By source file path |
| `files` | `db.delete_file()` | Removes the content-hash tracking row |
| Tantivy | `search.delete_symbols()` | Deletes by `Term::from_field_u64(f_id, symbol_id)` |
| petgraph | `graph.remove_file()` | `remove_node()` implicitly drops all edges |

---

## Lock ordering

To avoid deadlocks, locks are never held simultaneously across different stores
except where noted. The general pattern:

1. **db lock** — brief, released before touching graph/search
2. **graph + search lock** — acquired together only during bulk insert
3. **memory lock** — independent, never overlaps with the above

`reindex_file` is safe to call concurrently for different files. Concurrent calls
for the *same* file are serialized by the db lock and tested in
`concurrent_reindex_same_file`.
