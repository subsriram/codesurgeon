# Self-Improvement: Online Learning + Offline Eval

Status: **design, pre-implementation**. This doc proposes the architecture for
using call logs + observed code changes as a feedback signal to improve
codesurgeon over time. Two loops, one shared substrate.

- [1. Goals and non-goals](#1-goals-and-non-goals)
- [2. The signal](#2-the-signal)
- [3. Shared substrate](#3-shared-substrate)
- [4. Online loop — RL approaches](#4-online-loop--rl-approaches)
  - [4.1 Non-contextual bandits](#41-non-contextual-bandits)
  - [4.2 Contextual bandits](#42-contextual-bandits)
  - [4.3 Full RL (considered, not recommended)](#43-full-rl-considered-not-recommended)
  - [4.4 Learning-to-rank from implicit feedback](#44-learning-to-rank-from-implicit-feedback)
  - [4.5 Preference / RLHF-style](#45-preference--rlhf-style)
  - [4.6 Off-policy evaluation — the bridge to offline](#46-off-policy-evaluation--the-bridge-to-offline)
  - [4.7 Recommendation](#47-recommendation)
  - [4.8 Capsule shape as a learnable action — the v1.5 case](#48-capsule-shape-as-a-learnable-action--the-v15-case)
- [5. Offline loop](#5-offline-loop)
- [6. Guard rails](#6-guard-rails)
- [7. Configuration surface](#7-configuration-surface)
- [8. Observability](#8-observability)
- [9. Privacy and opt-in](#9-privacy-and-opt-in)
- [10. Roadmap](#10-roadmap)
- [11. Open questions](#11-open-questions)

---

## 1. Goals and non-goals

**Goals.**
- Turn every `run_pipeline` call into a datapoint that improves future ranking.
- Give dev iteration a *replay harness* so ranking PRs are judged on real user
  data, not only SWE-bench.
- Keep the whole thing local-first and opt-in for any sharing.
- Keep policy changes debuggable — no opaque neural end-to-end at step 1.

**Non-goals.**
- Training a model from scratch. Everything here reuses the existing ranker
  and tunes its parameters or reweights its outputs.
- Cross-user federated learning. Explicitly deferred to a later phase.
- Replacing SWE-bench. It stays as the synthetic correctness signal.

## 2. The signal

Every `run_pipeline` call returns a **capsule**: a list of (fqn, score) pairs
within a token budget, plus skeletons and pivots. Shortly after the call the
agent edits or reads some subset of those FQNs (or FQNs we failed to include).
That subset is our reward.

**Per-call reward** (`r ∈ [0, 1]`):

```
edited_set    = FQNs touched in [t, t + window_sec] after the call
returned_set  = FQNs in the capsule
hit_set       = edited_set ∩ returned_set
unused_set    = returned_set \ (edited_set ∪ read_set)

recall        = |hit_set| / max(1, |edited_set|)
rank_bonus    = mean(1 / rank_of(fqn)) for fqn in hit_set       # DCG-ish
precision_pen = |unused_set| / max(1, |returned_set|)           # share of capsule slots wasted

r_raw         = α · recall + (1 - α) · rank_bonus               # α ≈ 0.7
r             = clip( r_raw - λ · precision_pen, 0, 1 )         # λ ≈ 0.1 (small, tunable)
```

The precision penalty creates a gentle incentive for tighter capsules: an arm
that crams 20 FQNs into the budget and only "hits" 2 pays more than an arm
that returns 5 and hits 2. Default `λ = 0.1` is deliberately small — we want
the signal present from day 1 (so the column and config knob exist) but not
dominant until we've tuned it against real data. `λ = 0` is the escape hatch.

**Window semantics.** `window_sec = 600` (10 min) is the *observation window*
after a call — the span during which fs-events are attributed to that call.
Three outcomes possible:

- *Activity observed.* At least one edit (or read, if `capture_reads=true`)
  hits a workspace FQN during the window → the call becomes a datapoint,
  reward computed as above.
- *No activity observed.* Window closes with zero relevant events → call is
  **dropped, not scored zero**. Absence is ambiguous (user context-switched,
  went for coffee, abandoned the task) and training on it would bias the
  policy toward arms that happen to be selected when users are idle.
- *Overlapping calls.* If the user fires a second `run_pipeline` before the
  first window closes, both calls observe the same events. Acceptable — it
  slightly dilutes per-call attribution but is far cheaper than trying to
  partition events across overlapping windows.

Calls that never receive an outcome (either dropped or still within their
window) show up in `get_stats` as "reward coverage" — a health metric in its
own right.

**Known biases in this signal** (addressed in §6):
- *Selection bias*: we only learn about what we showed.
- *Causal ambiguity*: the agent edited X *because* we returned X, or *despite*
  a bad capsule.
- *Intent confound*: "fix" and "refactor" have different optimal rankings.

## 3. Shared substrate

Both loops depend on one SQLite file, `.codesurgeon/telemetry.db`:

```sql
CREATE TABLE calls (
  id            INTEGER PRIMARY KEY,
  ts            INTEGER NOT NULL,
  tool          TEXT NOT NULL,          -- run_pipeline | get_context_capsule | ...
  intent        TEXT,                    -- fix | refactor | explain | ...
  task_hash     TEXT NOT NULL,           -- blake3(task_text + local_salt)
  task_features BLOB,                    -- serialized feature vector (see §4.2)
  arm_id        INTEGER NOT NULL REFERENCES arms(id),
  capsule_fqns  TEXT NOT NULL,           -- JSON array, ranked
  scores        TEXT NOT NULL,           -- JSON array, same order
  token_count   INTEGER,
  latency_ms    INTEGER,
  commit_sha    TEXT,                    -- for offline replay
  workspace_id  TEXT NOT NULL            -- random stable ID
);

CREATE TABLE outcomes (
  call_id       INTEGER PRIMARY KEY REFERENCES calls(id),
  edited_fqns   TEXT NOT NULL,           -- JSON array
  read_fqns     TEXT,                    -- JSON array, optional (fs-read events)
  observed_at   INTEGER NOT NULL,
  window_sec    INTEGER NOT NULL,
  reward        REAL NOT NULL
);

CREATE TABLE arms (
  id            INTEGER PRIMARY KEY,
  name          TEXT UNIQUE NOT NULL,
  params_json   TEXT NOT NULL,           -- ranking policy params
  created_at    INTEGER NOT NULL,
  frozen        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE arm_stats (
  arm_id              INTEGER NOT NULL REFERENCES arms(id),
  stratum             TEXT NOT NULL DEFAULT '',   -- '' or intent label
  pulls               REAL NOT NULL DEFAULT 0,    -- REAL to allow decay
  reward_sum          REAL NOT NULL DEFAULT 0,
  reward_sq_sum       REAL NOT NULL DEFAULT 0,
  alpha               REAL NOT NULL DEFAULT 1,    -- Thompson Beta posterior
  beta                REAL NOT NULL DEFAULT 1,
  last_updated        INTEGER NOT NULL,
  PRIMARY KEY (arm_id, stratum)
);

CREATE TABLE counterfactual_outcomes (
  call_id       INTEGER NOT NULL REFERENCES calls(id),
  arm_id        INTEGER NOT NULL REFERENCES arms(id),
  reward        REAL NOT NULL,
  PRIMARY KEY (call_id, arm_id)
);
```

**Outcome collector.** An fs-events watcher runs in the MCP server process.
For each call, it buffers file events for `window_sec`, maps paths → FQNs via
the existing index, computes `r`, writes `outcomes`, and updates `arm_stats`.
Uses the existing `notify` crate pattern; one watcher per workspace.

**Arm registry.** Arms are ranking *policy variants*, not individual symbols.
A stock set lives in `crates/cs-core/src/policy.rs`; the user can add more via
`config.toml`. Each arm is a struct of ranker knobs:

```rust
struct PolicyArm {
    name: String,

    // Ranker knobs — how candidates are scored and fused.
    centrality_weight: f32,
    reranker_temp: f32,
    bm25_embed_mix: f32,        // 0 = pure BM25, 1 = pure embeddings
    graph_hop_budget: u32,
    recency_halflife_days: f32,

    // Capsule-shape knobs — how much of the ranked list is returned.
    // Includes the v1.5 adaptive-pivot schedule (see §4.8 below).
    pivot_schedule: PivotSchedule,

    // room to grow
}
```

Keep the live set small (4–8) so rewards accumulate fast. A `frozen=true` arm
is never selected by the policy but is always scored counterfactually — it's
the drift baseline.

## 4. Online loop — RL approaches

Ordered from simplest to most ambitious. Each section ends with a verdict:
**Adopt / Option / Defer**.

### 4.1 Non-contextual bandits

Treat arm selection as a multi-armed bandit with no per-call features. Three
standard policies.

#### ε-greedy

```
with prob ε     → uniform random arm
otherwise       → argmax_a (reward_sum[a] / pulls[a])
```

Plus exponential decay on `reward_sum` and `pulls` per `reward_halflife_days`
so the policy tracks drift.

- **Pros.** One knob, trivially debuggable, freeze by setting ε=0.
- **Cons.** Wastes exploration budget uniformly across arms; converges slowly
  when stratified (intent × arm).
- **Verdict.** **Adopt in Phase 1.** It's the right shape for the first live
  policy — you're still validating the reward signal itself.

#### UCB1

```
select argmax_a [ mean(a) + c · sqrt( ln(N) / n_a ) ]
```

- **Pros.** Deterministic, no randomness to seed, tighter regret bound than
  ε-greedy in stationary settings.
- **Cons.** Poor fit for non-stationary rewards (our setting: code changes,
  embeddings retrain). Needs a sliding-window variant (SW-UCB) to handle drift.
- **Verdict.** **Defer.** Not worth the extra complexity over ε-greedy.

#### Thompson sampling

Each arm has a `Beta(α, β)` posterior over reward probability (clip `r` to
[0, 1], which our recall metric already is).

```
select argmax_a [ sample from Beta(α_a, β_a) ]
update:   α += r;  β += (1 - r)
decay:    α, β ← α · e^(-Δt / τ), β · e^(-Δt / τ) + 1    # keep a mild prior
```

- **Pros.** Automatically allocates exploration toward uncertain arms; very
  sample-efficient; handles intent stratification cleanly (separate posterior
  per `intent × arm`).
- **Cons.** One more concept (posterior), prior choice matters a little.
- **Verdict.** **Adopt in Phase 3** (first policy of that phase). Swap-in
  after ε-greedy validates the reward signal. No schema change needed —
  `alpha`/`beta` columns already in `arm_stats`.

### 4.2 Contextual bandits

Use per-call features to pick the arm. This matters because the *best arm
depends on context*: "fix auth bug" and "rename every call site of X" want
very different rankings.

**Feature vector `x` per call.** Cheap to compute at call time:

```
intent:                 one-hot over {fix, refactor, explain, add, other}
task_text_embedding:    first 32 dims of the existing embedding model
has_diff:               bool (is this a get_diff_capsule?)
ws_languages:           bag-of-langs vector for the workspace
graph_fanout_hint:      log of touched-symbol degree in recent activity
time_since_last_call:   seconds, log-scaled
```

Total ~50 dims. Stored as `task_features` blob.

#### LinUCB / Disjoint Linear Thompson

Per arm, maintain `A_a ∈ R^{d×d}` and `b_a ∈ R^d`; estimate `θ_a = A_a^{-1} b_a`.

- LinUCB picks `argmax_a [ x^T θ_a + α · sqrt(x^T A_a^{-1} x) ]`.
- Linear Thompson picks `argmax_a [ x^T θ̃_a ]` with `θ̃_a ~ N(θ_a, A_a^{-1})`.

- **Pros.** Handles context without a deep model, closed-form updates, well-
  understood regret bounds. Fits in SQLite blobs (d=50 → 2.5 KB per arm).
- **Cons.** Linearity assumption; needs careful feature scaling. Sparse intents
  can destabilize `A_a^{-1}` early — ridge it.
- **Verdict.** **Adopt later in Phase 3** once flat Thompson shows gains.
  This is where real sample efficiency kicks in.

#### Neural contextual bandit

Small MLP maps `x → score_per_arm`. Trained with standard TD-free regression
against observed rewards; exploration via dropout-as-Bayesian or bootstrapped
heads.

- **Pros.** Captures non-linear interactions.
- **Cons.** Opaque, needs enough data, offline training step or periodic
  background retrain. Overkill until we have ≥10k calls.
- **Verdict.** **Defer.** Revisit only if LinUCB plateaus.

### 4.3 Full RL (considered, not recommended)

One could frame this as an MDP: state = workspace state, action = full
ranking, reward = downstream edit success, episodes = sessions. Policy gradient
(PPO) or Q-learning over ranked lists.

- **Why it's tempting.** Captures session-level credit assignment — a bad
  capsule now might cost later.
- **Why we won't.**
  - Action space is combinatorial (ranked lists of ~20 from ~10k FQNs).
  - Credit assignment across a session is genuinely hard and our reward signal
    is already noisy on a per-call basis.
  - No simulator. Real-user RL is slow and the variance is punishing.
  - Every known win over bandits here comes with interpretability cost.
- **Verdict.** **Defer indefinitely** unless bandit approaches hit a clear
  ceiling. The session structure can instead be captured by giving the
  contextual bandit a "recent activity" feature (§4.2).

### 4.4 Learning-to-rank from implicit feedback

A parallel track: use the logged data to train a ranker directly, offline.

- **LambdaMART / XGBoost ranker.** Train on `(query, returned_fqns,
  edited_fqns)` triples; pairwise or listwise loss against the edit signal.
  Output replaces or reweights the current reranker.
- **Listwise neural (e.g. small transformer)**: same data, more capacity,
  more care.

- **Pros.** Learns the ranking directly, not just which fixed policy to pick.
- **Cons.** Needs more data than bandit tuning; risk of overfitting the biased
  sample; harder to A/B safely.
- **Verdict.** **Option for phase 3.** Good complement to contextual bandits —
  bandit picks the *model*, LTR improves the *model's scores*.

### 4.5 Preference / RLHF-style

Whenever we log counterfactuals (§6), for each call we have arm A's capsule
and arm B's capsule, plus which one's FQNs overlapped more with the edits.
That's a pairwise preference `A ≻ B` or `B ≻ A`, usable for:

- **DPO-style tuning** of a scoring model.
- **Reward-model fitting** if we ever go down the full-RL road.

- **Pros.** Turns sparse real-valued rewards into dense pairwise preferences.
- **Cons.** Preferences inherit the same selection bias unless counterfactual
  logging is broad.
- **Verdict.** **Option for phase 4.** Cheap to *collect* from day one — we
  should start logging the pairs even if we don't train on them yet.

### 4.6 Off-policy evaluation — the bridge to offline

Inverse Propensity Scoring (IPS) and Doubly Robust (DR) let us estimate "what
would the reward have been under policy π' given data collected under π?"
This is the glue that makes the offline replay harness (§5) trustworthy.

- Store per-call `selection_prob[chosen_arm]` from the policy (ε-greedy and
  Thompson both yield this).
- At eval time, weight each logged reward by `π'(a|x) / π(a|x)`.
- **Verdict.** **Adopt alongside Phase 1.** It's a schema detail (one extra
  REAL column, `selection_prob`) but unlocks honest offline comparison.

### 4.7 Recommendation

Policy progression, anchored to roadmap phases (§10):

| Phase | Policy | Rationale |
|-------|--------|-----------|
| Phase 0 | *none* (logging only) | Collect data, validate reward signal |
| Phase 1 | ε-greedy, non-contextual, intent-stratified | One knob, fully debuggable |
| Phase 2 | (unchanged) — add guard rails + offline replay | Make the loop honest before scaling it |
| Phase 3 | Thompson sampling → Linear Thompson | Sample efficiency; contextual features |
| Phase 4a *(optional)* | LambdaMART reranker as a new arm | Train the scorer itself, let bandit choose |
| Phase 4b *(optional)* | DPO on pairwise counterfactuals | Learn from logged `A ≻ B` pairs |

Full RL (PPO, Q-learning over ranked lists) is **not on the path** — see §4.3.

### 4.8 Capsule shape as a learnable action — the v1.5 case

The adaptive pivot cap shipped in [`docs/explicit-symbol-anchors.md`](explicit-symbol-anchors.md#v15--adaptive-pivot-cap-based-on-anchor-confidence-landed)
is a hand-crafted contextual policy: given a context (`AnchorStats` — how many
anchor hits were exact, how many were BM25 fallback, how many distinct source
files), it picks an action (`effective_pivots ∈ {3, 5, 8}`) via a three-bucket
rule:

```
CLEAN    (≥3 exact, 0 fuzzy, ≥2 distinct source files) → 3 pivots
MEDIUM   (≥1 exact OR any fuzzy)                        → 5 pivots
DEFAULT  (no anchor fires)                              → 8 pivots
```

This is exactly the shape of problem the self-improvement loop is designed
to handle. Three observations make it a near-ideal first target:

1. **The reward already scores this correctly.** Precision penalty
   (§2) penalises wasted capsule slots, while recall rewards keeping enough
   pivots to cover the edited FQNs. Over-capping hurts recall; under-capping
   hurts precision. No new reward machinery needed.
2. **The context is already computed.** `AnchorStats` is returned from
   `anchor_candidates`; we just need to log it alongside each call.
3. **The v1.5 rule gives a strong seed arm.** The hand-tuned thresholds are
   a reasonable prior — we don't start from scratch.

#### Pivot policy as an arm parameter

Capsule shape is a *separate action dimension* from ranker parameters. We
encode it as `PivotSchedule` on `PolicyArm`:

```rust
enum PivotSchedule {
    /// Fixed cap regardless of context. Useful as a sanity-check arm.
    Fixed(usize),
    /// v1.5-style bucketed rule over AnchorStats. Thresholds are data.
    Bucketed {
        clean:   usize,   // 3 in v1.5
        medium:  usize,   // 5 in v1.5
        default: usize,   // 8 in v1.5
    },
    /// Phase 3 only: learned mapping from context features to pivot count.
    Learned { model_id: String },
}
```

#### Per-phase treatment

- **Phase 0.** Log `AnchorStats` and the chosen `effective_pivots` on every
  call. Add two columns to `calls` (or serialise into `task_features`):
  `anchor_stats_json`, `effective_pivots`. No behavior change.
- **Phase 1 (ε-greedy).** Seed a few `PivotSchedule::Bucketed` arms:
  - `baseline_v15` (frozen): `{clean:3, medium:5, default:8}` — the shipped rule.
  - `aggressive`: `{clean:2, medium:4, default:6}`
  - `conservative`: `{clean:5, medium:7, default:10}`
  - `flat_5` (`Fixed(5)`): ignores anchor stats — counterfactual sanity check.

  Validates whether v1.5's thresholds are actually best.
- **Phase 3 (contextual).** Add `AnchorStats` to the feature vector
  (`resolved_exact`, `resolved_bm25_name`, `distinct_source_files`,
  `anchors_extracted`). Linear Thompson now learns
  `pivot_count ← f(anchor_stats, intent, workspace_langs, …)` directly.
  The bucketed rule becomes learned decision boundaries, and those boundaries
  can differ per intent (a "fix" may want fewer pivots than a "refactor").
  Realised as `PivotSchedule::Learned` — the model is fit offline from logs,
  same pattern as the Phase 4a LambdaMART reranker.

#### What this pattern generalises

v1.5 is the first of several capsule-shape knobs that want learning:

- **Number of pivots** — covered above.
- **Number of skeletons** — fixed at 20; similarly context-dependent.
- **Token budget fraction spent on skeletons vs pivot bodies** — currently
  implicit in the pivot/skeleton count ratio.
- **Whether to include adjacent symbols** — boolean action.

All of these follow the same recipe: context = features we already compute,
action = a small discrete or scalar parameter, reward = the same §2 reward.
Treat §4.8 as the template; we don't need a new doc each time.

## 5. Offline loop

### Replay harness

`cargo run -p cs-eval --bin replay -- <telemetry.db> --policy <name>`

For each logged call:

1. Check out `commit_sha`, rebuild or load the index snapshot.
2. Re-run the ranker under the candidate policy.
3. Score the new capsule against the logged `edited_fqns`.
4. Optionally compute IPS-weighted delta vs. the logged policy.

Output: `recall@k`, `MRR`, `token_budget_efficiency`, per-intent breakdowns,
and a CSV for notebook analysis.

### Integration with CI

Every PR that touches `engine.rs`, `search.rs`, `graph.rs`, or policy params:

- Required check: SWE-bench pass@1 delta (existing).
- Required check: replay recall@k delta on a frozen telemetry corpus.

Two numbers, two failure modes covered.

### Telemetry export

`codesurgeon telemetry export --out export.ndjson` emits:

```json
{"ts_bucket":"2026-04","intent":"fix","task_hash":"<blake3>",
 "features":[...],"capsule":["a::b","c::d"],"edited":["a::b"],
 "arm":"ucb_c_1_5","selection_prob":0.62,"reward":0.5,"commit":"<sha>"}
```

- Task text is hashed (salted per workspace, salt stays local).
- FQNs are path-normalized; can be fully redacted via `--redact-fqns`.
- User inspects the file before sharing.
- No auto-upload in this design. If cross-user aggregation ever ships, it's a
  separate RFC.

## 6. Guard rails

- **Counterfactual logging.** On X% of calls (`counterfactual_sample_rate`,
  default 0.05, tunable), run every arm's ranker over the same query and log
  what each would have returned. Score all of them against the same observed
  edits. Writes to `counterfactual_outcomes`. This:
  - breaks the selection-bias feedback loop,
  - feeds pairwise preferences (§4.5),
  - enables honest DR estimates (§4.6).
  Bounded cost: ranking is fast and we gate by sample rate.

- **Frozen baseline arm.** Never updated, never selected by the policy, but
  always scored counterfactually. If all learned arms fail to beat it over N
  days, surface a warning via `get_stats` and auto-bump exploration.

- **Per-arm daily reward delta cap.** One noisy week can't crater a variant.

- **Minimum pulls before exploit.** Prevents early lucky arms from locking in.

- **Reward auditing.** A `codesurgeon telemetry audit` CLI samples 20 recent
  outcomes and pretty-prints `(task, capsule, edited, reward)` so the
  developer can sanity-check the signal manually. Non-optional: the reward
  function is the most load-bearing assumption in the whole design.

- **Kill switch.** `exploration.policy = "off"` bypasses the bandit entirely
  and always selects the named `default_arm`. Logging continues. Single-knob
  rollback.

## 7. Configuration surface

New section in `.codesurgeon/config.toml`:

```toml
[telemetry]
enabled            = true        # master switch
window_sec         = 600
capture_reads      = false       # fs-read events, more noise
db_path            = ".codesurgeon/telemetry.db"
salt_path          = ".codesurgeon/telemetry.salt"

[exploration]
policy             = "epsilon_greedy"   # off | epsilon_greedy | thompson | lin_thompson
default_arm        = "baseline"
epsilon            = 0.10
reward_halflife_days = 14
min_pulls_before_exploit = 30
intent_stratified  = true
counterfactual_sample_rate = 0.05
per_arm_daily_reward_cap = 5.0
precision_penalty_lambda = 0.10        # 0.0 disables the unused-entry penalty

[[exploration.arms]]
name = "baseline"
frozen = true
centrality_weight = 1.0
reranker_temp = 1.0
bm25_embed_mix = 0.5
graph_hop_budget = 2

[[exploration.arms]]
name = "high_graph"
centrality_weight = 1.5
bm25_embed_mix = 0.35
graph_hop_budget = 3

# ... more arms
```

All defaults chosen so telemetry *collection* is on but the policy is
conservative. Disable collection entirely by `enabled = false`.

## 8. Observability

Extend `get_stats` with a `self_improvement` section:

```
arms:
  baseline (frozen)        pulls=842  r̄=0.41±0.02  p_select=0.00
  high_graph               pulls=611  r̄=0.48±0.03  p_select=0.34  ↑
  deep_rerank              pulls=580  r̄=0.46±0.03  p_select=0.28
  embed_heavy              pulls=490  r̄=0.39±0.04  p_select=0.12  ↓

drift: learned arms +0.07 vs baseline (14-day window) ✓
reward coverage: 73% of calls received an outcome within window
counterfactual calls: 4.8% (target 5%)
```

New MCP tool `get_learning_report` (optional) returns the same structured.

## 9. Privacy and opt-in

- Telemetry is **on by default, local-only**. No network traffic.
- `task_hash` uses a per-workspace random salt stored in
  `.codesurgeon/telemetry.salt` with mode `0600`. Rotating the salt severs
  linkability across time buckets.
- `telemetry export` is explicit and manual.
- Add `.codesurgeon/telemetry.db` and `.salt` to the default `.gitignore`
  template generated by `codesurgeon init`.
- Document clearly in README — one section, not buried.

## 10. Roadmap

**Terminology.** *Phase N* is a roadmap milestone — a block of work with one
exit criterion. It breaks into *PR N.M* units listed under "PR breakdown".
Policies (ε-greedy, Thompson, Linear Thompson, …) are referred to by name,
never as "v1/v2/..."; each phase's policy column in §4.7 shows which policy
ships where.

Every phase must ship behind a config flag and leave the default user
experience unchanged until it's proven.

### Phase 0 — Foundation (1–2 weeks)

**Goal.** Start collecting high-quality `(call, outcome, reward)` tuples on
real workspaces. No policy yet — the existing ranker runs unchanged.

**Design points.**
- Telemetry lives in a dedicated crate `cs-telemetry` so the schema, writer,
  and collector are testable in isolation from MCP plumbing.
- The outcome collector is a single background task per workspace. It listens
  to fs-events via `notify`, buffers path events in a time-bounded ring, and
  on window close does the path → FQN mapping using the existing symbol index
  (cached in memory, no DB round-trip per event).
- Writes to `telemetry.db` are batched on a 1s timer to avoid hot-path latency
  on `run_pipeline`. A bounded `mpsc` channel from the writer API to the DB
  task absorbs bursts; overflow drops new events with a counter (visible in
  `get_stats`) rather than blocking callers.
- The precision-penalty `λ` is wired through from day 1, defaulting to 0.1.
- Salt file created on first run with mode 0600.

**PR breakdown.**
- **PR 0.1** — `cs-telemetry` crate skeleton: schema SQL, `sqlx` or `rusqlite`
  migrations, integration test that opens/migrates/closes a scratch DB.
- **PR 0.2** — `TelemetryWriter` API (`log_call`, `log_outcome`) + batch task
  + overflow counter. Unit tests with a fake clock.
- **PR 0.3** — fs-events collector: path → FQN mapper, windowing logic,
  reward function (including precision penalty), bounded event dedup.
  Integration test that synthesizes a call + simulated fs events and asserts
  reward matches hand-computed value.
- **PR 0.4** — Wire `log_call` into `run_pipeline` in `cs-mcp`. Feature-gated
  by `[telemetry] enabled`. Logs the already-computed `AnchorStats` and
  `effective_pivots` alongside the capsule (see §4.8) — no behavior change
  to v1.5, just observation.
- **PR 0.5** — `codesurgeon telemetry audit` CLI: prints the last N calls
  with capsule, edits, and reward side-by-side.
- **PR 0.6** — `.gitignore` template updates, README section, docs link.

**Exit criterion.** 100 real calls logged on the codesurgeon dev workspace
over a week; developer runs `telemetry audit` and agrees the rewards look
sane for at least 20 of them. One open question: if audit reveals the reward
is noisy in a surprising way, fix the formula *before* Phase 1 — the bandit
is only as good as the signal.

### Phase 1 — ε-greedy online (1 week)

**Goal.** A live bandit picking between arms, with a hard kill switch.

**Design points.**
- Arms are loaded from `[[exploration.arms]]` at MCP start. Unknown arm
  names in logs (old configs) are treated as opaque — we never crash on
  historical data.
- Intent stratification: when on, every `(arm, intent)` pair is its own row
  in `arm_stats`. Unknown intents fall into stratum `''` (global).
- Decay is *lazy*: on every read of `arm_stats`, we compute the time elapsed
  since `last_updated` and scale `pulls` / `reward_sum` in-place. No cron.
- ε-greedy selection is deterministic given a seeded RNG — makes tests
  reproducible. In production the seed comes from `SystemTime`.
- `selection_prob[chosen_arm]` stored on every call (`ε / n_arms` for
  random picks, `1 - ε + ε/n_arms` for the greedy pick). Needed for IPS.

**PR breakdown.**
- **PR 1.1** — Arm registry crate-module + config loader + baseline/frozen
  semantics, including `PivotSchedule` (§4.8). Tests for config parse errors.
- **PR 1.2** — `arm_stats` reader with lazy decay. Property test: two reads
  with no writes between them return identical decayed values.
- **PR 1.3** — ε-greedy policy (seeded RNG), `selection_prob` computation,
  wiring into both the ranker's arm-parameter application path *and* the
  `effective_pivots` selection (replacing v1.5's hand-coded match with the
  chosen arm's `PivotSchedule`).
- **PR 1.4** — Intent stratification toggle.
- **PR 1.5** — `get_stats` `self_improvement` section.
- **PR 1.6** — Kill switch (`policy = "off"`) integration test.

**Exit criterion.** On the dev workspace, one non-baseline arm beats the
baseline by ≥3% recall@k over 2 weeks of normal use. If nothing beats
baseline, *do not* proceed — either the arm set is too narrow or the reward
signal is wrong; iterate on those first.

### Phase 2 — Guard rails + offline replay (1–2 weeks)

**Goal.** Make the online loop honest and unlock the offline eval harness
that every future ranking PR will depend on.

**Design points.**
- Counterfactual computation reuses the ranker's main path with arm
  parameters swapped; it runs synchronously on the call thread only when a
  uniform RNG draw < `counterfactual_sample_rate`. For the remaining 95% of
  calls, there's zero overhead.
- The "frozen baseline" arm is always scored counterfactually regardless of
  sample rate — it's our drift detector and must not be starved of data.
- `cs-eval` is a new crate so replay doesn't drag MCP dependencies.
- The CI replay corpus is a curated, committed `.ndjson` file at
  `bench/replay_corpus.ndjson` — not user data. ~200 entries drawn from dev
  workspaces, hand-reviewed. Ranking PRs are scored against this corpus,
  not against whatever the local machine happens to have.
- IPS weights are *clipped* at a configurable ceiling (default 10) to avoid
  one rare (ε-selected) call dominating the estimate.

**PR breakdown.**
- **PR 2.1** — Counterfactual sampler + `counterfactual_outcomes` writer.
  Test: sample rate 1.0 yields one row per (call, arm).
- **PR 2.2** — Frozen-baseline drift alarm (surfaced in `get_stats`; if all
  learned arms tie or underperform over N days, auto-bump `ε`).
- **PR 2.3** — `cs-eval` crate skeleton + index snapshot loader by SHA.
- **PR 2.4** — `replay` binary: given a telemetry DB and a policy spec,
  emits recall@k/MRR/token-efficiency per intent to stdout JSON.
- **PR 2.5** — IPS + DR weighting in replay, with clipping.
- **PR 2.6** — Committed replay corpus + a CI workflow that runs replay on
  it and comments the recall@k delta on ranking-touching PRs.
- **PR 2.7** — Document the replay workflow in `docs/ranking.md` so future
  ranking changes follow it.

**Exit criterion.** A deliberately-regressive ranking PR is submitted as a
test; CI flags the replay delta and fails the check.

### Phase 3 — Thompson + contextual bandits (2–3 weeks)

**Goal.** Better sample efficiency, especially across intent strata.

**Design points.**
- `policy = "thompson"` is a pure swap — the `alpha`/`beta` columns already
  exist, and decay applies to them the same way.
- Feature extraction lives in `cs-telemetry::features`. Versioned: each
  feature vector stores a `feature_schema_version` byte. Replay refuses to
  mix versions; when we bump the version, we start a new corpus.
- For LinUCB/Linear Thompson we store per-arm `A` and `b` as BLOBs in a new
  `arm_linear_state` table. Updates are in-place and incremental (Sherman-
  Morrison for the inverse) so we never rebuild from scratch.
- Ridge parameter `λ_ridge` (not to be confused with precision penalty λ) is
  configurable; default 1.0 handles cold-start reasonably.

**PR breakdown.**
- **PR 3.1** — Thompson policy implementation + tests (Beta posterior,
  temporal decay).
- **PR 3.2** — Feature extractor (intent one-hot, task embedding, workspace
  langs, fanout hint, time-since-last-call) + schema versioning.
- **PR 3.3** — `arm_linear_state` table + Sherman-Morrison update.
- **PR 3.4** — LinUCB and Linear Thompson policies behind the same interface.
- **PR 3.5** — Feature-importance display in `get_stats` (coefficient
  magnitudes per arm).

**Exit criterion.** On replay of accumulated logs, Linear Thompson beats
flat Thompson by ≥2% recall@k, with the gain concentrated in minority
intents (where per-stratum data was sparsest).

### Phase 4 — Optional extensions

Pursue only if earlier phases plateau or demand emerges. Each is its own
mini-RFC before PRs.

- **Phase 4a.** LambdaMART / XGBoost reranker trained on accumulated logs.
  Ship as a new arm so the bandit decides when it wins. Requires an offline
  training job (nightly or on-demand).
- **Phase 4b.** DPO on pairwise counterfactual preferences. Tune a scoring
  model from `(winning_capsule, losing_capsule)` pairs already being logged
  since Phase 2.
- **Phase 4c.** Opt-in cross-workspace aggregation. Separate RFC because
  privacy/consent/storage questions dominate the engineering ones.

## 11. Open questions

1. **Read events as signal.** Do we count "file opened in editor" as weak
   reward? Noisier than edits but higher coverage. Start off, revisit.
2. **Stratum granularity.** Intent (5 values) or intent × language? Latter
   fragments data fast.
3. **Arm authoring UX.** Hand-edit config vs. a `codesurgeon arms new` CLI
   that scaffolds a variant from a baseline.
4. **Embedder retrains.** If the embedder changes, old feature vectors aren't
   comparable. Feature schema is versioned (Phase 3) and replay refuses to
   mix — but we haven't decided whether old calls become unusable or get
   re-featurized lazily.

**Resolved during design review (2026-04-17):**
- *Precision penalty for unused capsule entries.* Adopted with small default
  `λ = 0.1`, tunable in config. Goal: nudge toward tighter capsules without
  dominating the reward signal until we have data to tune it.
- *Window length.* 600s fixed for Phase 0. Not adaptive yet — revisit if
  coverage metrics suggest users have systematically shorter/longer sessions.
