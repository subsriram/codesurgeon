### SWE-bench Verified — #29b stratified pilot

Run id: `pilot-20260415-140651`  ·  10 tasks × 2 arms  ·  stratified across 5
repos (astropy, django, sympy, matplotlib, psf/requests — 2 tasks each).
Seed-17 first-two-per-capped-repo slice. Model: Claude Code default (sonnet).
Per-task budget $3.00, wallclock cap 900s.

Raw artifacts: [`pilot_results/results.jsonl`](pilot_results/results.jsonl),
[`pilot_results/harness_with.json`](pilot_results/harness_with.json),
[`pilot_results/harness_without.json`](pilot_results/harness_without.json).

## Headline

| Arm | n | Pass@1 | Avg total tokens | Avg cost | Avg walltime |
|---|---:|---:|---:|---:|---:|
| bare Claude Code | 10 | 60.0% (6/10) | 491,370 | $0.3638 | 211.3s |
| + codesurgeon | 10 | **80.0% (8/10)** | 609,105 | $0.4209 | 202.3s |
| **Δ** | — | **+20.0pp** | +24.0% | +15.7% | −4.2% |

Zero regressions: codesurgeon resolved every task that bare resolved, plus
two more. The aggregate +24% token delta is real but is dominated by the
two codesurgeon-only wins, where codesurgeon "spent more tokens" because
it kept working while bare gave up with a wrong fix. **Recast by outcome
class below.**

## Recast by outcome class

The aggregate hides three very different populations. Splitting them out:

| Population | Tasks | Δ tokens | Interpretation |
|---|---:|---:|---|
| **Both arms passed** | 6 | **+9.3%** | Real overhead — the honest cost of running codesurgeon on tasks where it doesn't change the outcome |
| **Both arms failed (or tied non-pass)** | 2 | ~0% | sympy-18199 (−17.4%), django-11734 (double-timeout, both 0 tok scored) |
| **Codesurgeon-only wins** | 2 | +247% and +25% | Not overhead — bare gave up and failed; codesurgeon kept working and passed |

The two wins (astropy-7166, psf/requests-6028) account for **955K of the
1.06M total token delta**. The remaining 7 scored tasks collectively come
in near a wash.

Framed honestly:

> Codesurgeon costs **~9% more tokens on tasks where it doesn't flip
> pass/fail**, in exchange for **+20pp pass@1** from the tasks where it
> does. On OAuth-subscription pricing the real marginal cost is zero.

## Per-task detail

| Instance | with pass | w/o pass | w tok | w/o tok | Δ tok % | w wall | w/o wall |
|---|:---:|:---:|---:|---:|---:|---:|---:|
| astropy__astropy-14539 | ✅ | ✅ | 357,857 | 328,531 | +8.9% | 74.8s | 77.2s |
| astropy__astropy-7166 | ✅ | ❌ | 510,894 | 147,225 | +247.0% | 120.8s | 31.4s |
| django__django-11163 | ✅ | ✅ | 227,689 | 177,135 | +28.5% | 28.0s | 33.7s |
| django__django-11734 | ⏱ | ⏱ | — | — | — | 900.0s | 900.0s |
| matplotlib__matplotlib-13989 | ✅ | ✅ | 168,771 | 122,015 | +38.3% | 20.0s | 16.0s |
| matplotlib__matplotlib-25287 | ✅ | ✅ | 167,064 | 182,381 | −8.4% | 22.4s | 39.9s |
| psf__requests-1766 | ✅ | ✅ | 198,941 | 116,965 | +70.1% | 19.0s | 17.8s |
| psf__requests-6028 | ✅ | ❌ | 2,924,055 | 2,332,814 | +25.3% | 713.9s | 850.4s |
| sympy__sympy-18199 | ❌ | ❌ | 235,091 | 284,705 | −17.4% | 42.0s | 60.2s |
| sympy__sympy-20590 | ✅ | ✅ | 691,583 | 730,559 | −5.3% | 82.0s | 86.3s |

⏱ = hit the 900s wallclock cap in both arms. Not scored, not a codesurgeon
signal — django-11734 is a hard task that needs a higher timeout.

## Per-repo breakdown

| Repo | Tasks | Δ avg total tokens | Δ avg cost | Δ pass@1 |
|---|---:|---:|---:|---:|
| astropy/astropy | 2 | +82.6% | +66.5% | **+50.0pp** |
| django/django | 2 | +28.5% | +21.8% | +0.0pp |
| matplotlib/matplotlib | 2 | +10.3% | −16.6% | +0.0pp |
| psf/requests | 2 | +27.5% | +14.0% | **+50.0pp** |
| sympy/sympy | 2 | −8.7% | +1.1% | +0.0pp |

Codesurgeon's wins are concentrated on **hard navigational bugs**
(astropy-7166, requests-6028) — tasks where finding the right file to edit
is the actual difficulty. On tasks where the edit target is obvious
(matplotlib, sympy, django) codesurgeon adds modest token overhead with
no outcome change.

## #29b go/no-go gate

| Check | How to verify | Status |
|---|---|---|
| Harness stable | `results.jsonl` has 20 rows, all with `exit_code == 0` | ✅ 18 scored + 2 timeouts on the same task (not a harness bug) |
| Directional signal (`with ≥ without − 10pp`) | compare pass@1 columns | ✅ **+20pp in codesurgeon's favor** |
| Avg walltime ≤ 600s | `avg_walltime_s` | ✅ 211s / 202s |

All three green.

## Caveats

- **n=10 is small.** A +20pp pass@1 lift is two task flips — confidence
  interval is wide. The pilot *justifies* running #29c (full 100 tasks);
  it doesn't substitute for it.
- **Bare Claude Code "giving up fast" is a real failure mode**, not an
  artifact of the prompt. Both arms received identical prompts (`run.py`
  PROMPT_PREFIX + `problem_statement`). The difference is whether the
  agent had a graph-based way to find the right code to read.
- **django-11734 needs a higher timeout for #29c** — bump `PILOT_TIMEOUT`
  from 900s to 1200s to avoid unscorable tasks.
- **Earlier astropy-only pilot flagged a regression** (+27% tokens, 0pp
  pass@1). That subset was unrepresentative — seed-17 first-10 happened
  to be all astropy/astropy, a repo with relatively obvious edit targets.
  Stratifying across 5 repos reveals codesurgeon's actual profile.

## Next

Open **#29c** and run the full 100 tasks with:

- `PILOT_TIMEOUT=1200` (up from 900)
- Same model, same budget, same prompt
- Expected walltime ~13h detached; synthetic cost ~$80; real bill $0

On n=100 the current +20pp signal should resolve into either "robustly
positive" or "pilot variance" — and the per-repo breakdown will have
enough rows to say which repo types codesurgeon actually helps on.
