# Design (stub): Strip docstring examples from call-edge extraction

> **Status**: stub — no implementation yet.
> **Target**: `crates/cs-core/src/indexer.rs` (per-language call-extraction
> walks).
> **Related**: `docs/explicit-symbol-anchors.md` (anchors), `docs/query-history-ranking.md` (memory channel).

## Motivation

The indexer extracts `calls` edges by walking each function/method body and
recording every identifier that resolves to a known symbol. The walk
**includes the docstring**, so identifiers mentioned in docstring example
blocks (`>>> ...`) get treated as call sites.

When the resolver can't pin a docstring identifier to a single target (e.g.
`evalf` exists on dozens of classes), it falls back to creating an edge to
**every symbol with that name**. The graph fills up with edges that have no
relationship to the function's actual behavior.

## Concrete evidence

From `target/swebench/with/sympy__sympy-21612/post_run_index.db` (sympy
v1.10.dev0 indexed by codesurgeon v1.6):

The dispatcher `sympy/parsing/latex/__init__.py::parse_latex` is a 30-line
function whose body literally calls only `import_module` and (dynamically)
`_latex.parse_latex`. **Two calls.**

Yet the graph records **22 outgoing `calls` edges** for it:

```
Subs::evalf, MinMaxBase::evalf, evalf, IdealSolitonDistribution::dict,
NegativeInfinity::evalf, Point::evalf, lib_interval::sqrt,
FiniteDomain::dict, BinomialDistribution::dict, HolonomicFunction::evalf,
EvalfMixin::evalf, BasisDependent::evalf, FiniteDensity::dict,
SingleFiniteDistribution::dict, doctest_depends_on,
DiscreteUniformDistribution::dict, MatrixOperations::evalf,
miscellaneous::sqrt, FiniteDistributionHandmade::dict, Infinity::evalf,
import_module, _parse_latex_antlr.py::parse_latex
```

The 20 spurious edges trace one-to-one to identifiers in the docstring
example block:

```python
>>> expr = parse_latex(r"\frac {1 + \sqrt {a}} {b}")
>>> expr
(sqrt(a) + 1)/b
>>> expr.evalf(4, subs=dict(a=5, b=2))
1.618
```

(`evalf`, `dict`, `sqrt` each match dozens of unrelated symbols → the
resolver fans out to all of them.)

## Downstream impact

1. **Skeleton noise.** `select_adjacents` walks `graph.dependencies(pivot)`
   to choose skeleton neighbors. With contaminated edges, skeletons become
   "every method named `evalf`" instead of "structurally related code".
   The sympy-21612 capsule had 17 skeletons; 15 were doctest-token symbols
   with zero relationship to LaTeX parsing.
2. **Centrality inflation.** Hub symbols like `evalf`, `sqrt`, `dict`
   accumulate massive in-degree from docstring mentions. `centrality_score`
   then over-promotes them in BM25 + centrality re-ranking.
3. **Impact-graph rot.** `get_impact_graph` claims a docstring-only mention
   "depends on" the cited symbol. False blast-radius warnings on refactors.
4. **Cross-codebase scale.** Every docstring-heavy codebase suffers
   (sympy, numpy, scipy, matplotlib, sklearn, pandas, …). Codebases that
   value example-driven docstrings are precisely the ones where this hurts
   the most.

## Fix sketch

In the per-language call-extraction walk, identify the docstring node of
each function/method and **skip it** during identifier collection.

For Python (tree-sitter), the docstring is the first child of the function
body when that child is an `expression_statement` containing a `string`
node. Two granularities:

- **Coarse:** skip the entire docstring node.
- **Fine:** keep the prose but skip lines starting with `>>> ` (and their
  `... ` continuation lines + the indented output line that follows). Lets
  prose-mentioned identifiers stay (they sometimes are real collaborators),
  but drops the doctest example noise.

Coarse is one node-skip in the AST walk, ~10 lines of Rust per language.
Fine adds a regex pre-pass to the docstring text. Recommend coarse first
(simpler, lower-risk, larger effect) and add fine only if losing prose
identifiers shows up as a regression.

Same pattern applies to:
- Rust: `///` doc comments above an item, `//!` module-level.
- TypeScript / JavaScript: leading `/** */` JSDoc.
- Swift: `///` doc comments.

For each language, locate where the indexer collects identifiers from a
function body, and exclude the docstring/doc-comment nodes from that walk.

## Validation plan

- [ ] Unit test in `crates/cs-core/src/indexer.rs` (per language): index a
      synthetic file containing a function whose docstring mentions
      `OtherSymbol`; assert `OtherSymbol` does **not** appear in the
      function's outgoing `calls` edges.
- [ ] Integration test: index a snippet whose docstring is a doctest with
      `>>> foo.evalf()`; assert no `calls` edge to any `evalf` symbol.
- [ ] Re-index the SWE-bench sympy/matplotlib/xarray warm caches; compare
      pre/post edge counts. Expect significant reduction on docstring-heavy
      modules.
- [ ] Re-run sympy-21612: confirm the dispatcher now has 2 outgoing edges
      (`import_module`, `_parse_latex_antlr.py::parse_latex`) and the
      capsule's skeleton list is dominated by structurally-related symbols.
- [ ] Centrality regression check: assert no symbol's centrality drops by
      more than X% on representative codebases (some loss is expected and
      good — that's the point — but a cliff would suggest over-stripping).

## Risks

1. **Lost real signal.** Some docstrings deliberately reference real
   collaborators in prose ("delegates to `BarHelper.process`"). Coarse
   stripping loses that. Mitigation: ship coarse first, measure recall on
   a fixed evaluation set; if regressions appear, switch to fine.
2. **Tree-sitter docstring detection varies by language.** Each parser may
   structure docstring nodes differently. Have to test per-language and
   add a fallback (skip nothing) for unknown shapes rather than crash.
3. **Existing indexes are dirty.** Need a re-index for the fix to take
   effect. Add a `manifest.json` schema bump so existing workspaces force
   a fresh index next time `codesurgeon index` runs.
4. **Embedding cache.** Symbol embeddings are computed from the body text,
   not the edges. Stripping doctests from edges does not touch embeddings.
   No re-embed required, but worth verifying.

## Out of scope

- Stripping doctest tokens from BM25 body field (a separate, larger
  question covered by considering how much docstring content should
  contribute to retrieval at all).
- Doctest extraction as a *positive* signal (e.g. "this function is
  documented with example X" → boost when query mentions X). That's
  follow-up work; first stop the negative noise.
