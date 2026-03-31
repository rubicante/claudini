# Agent Log: claude_my_run

**Goal:** Break Qwen2.5-7B on random strings under 1e15 FLOPs
**Config:** `random_train_q4` (Qwen/Qwen2.5-7B-Instruct, 1e15 FLOPs, 4-bit, samples 0-2, seed 0)
**Branch:** `loop/my_run`
**Method prefix:** `claude_my_run_v`

## Baseline

Best method at 1e15 FLOPs (random_train_q4, Qwen2.5-7B): **PGD at 11.33** (mean over 5 samples).
PGD gets ~590 gradient steps at 1e15. GCG gets only ~4 steps — discrete search too expensive.

## v1 — PGD + LSGM(0.85) + stripped auxiliary losses ✓ BEST

**Issue:** [#15](https://github.com/rubicante/claudini/issues/15)
**Results:** s0=11.76, s1=7.10, s2=6.73 → **mean=8.53** (+2.80 vs PGD)

LSGM + stripped aux losses dramatically helps samples 1&2 but NOT sample 0.
Note: v1 is actually WORSE than PGD on sample 0 (11.76 vs 11.33).
Patience mechanism never fires — relaxed loss always improves.

## v2 — LSGM(0.75) + noisy patience escape (scale=1.5)

**Issue:** [#16](https://github.com/rubicante/claudini/issues/16)
**Results:** s0=11.51, s1=8.40, s2=6.63 → mean=8.85 (worse than v1)
gamma=0.75 disrupted sample 1's trajectory. Large noise scale too disruptive.

## v3 — LSGM(0.85) + gentle noise kick (scale=0.5) — identical to v1 in practice

**Issue:** [#17](https://github.com/rubicante/claudini/issues/17)
**Results:** s0=12.05, s1=7.10, s2=9.11 → mean=9.42 (worse than v1)
Patience never fires → noise kicks never triggered. Variance = GPU non-determinism.
3 samples insufficient for reliable comparison — results vary per GPU run.

## v4 — K=2 restarts + LSGM(0.85)

**Issue:** [#18](https://github.com/rubicante/claudini/issues/18)
**Results:** s0=11.30, s1=10.23, s2=11.02 → mean=10.85 (much worse than v1)
K=2 gives only ~295 steps per restart. Breakthrough for s1&s2 happens at step 400-590.
With 295 steps, breakthroughs don't happen. More steps > more restarts at this budget.

## v5 — Stripped losses only, NO LSGM — ablation

**Key question**: Does LSGM help or hurt sample 0?
v1 LSGM makes s0=11.76 (vs PGD=11.33 without LSGM). Is LSGM hurting s0?

**Issue:** (pending)
**Results:** (pending)

## What to try next (v6)

Ablation results will determine:
- If v5 > v1: LSGM hurts sample 0 → adaptive LSGM (activate only after step 200)
- If v5 ≈ v1 (same 1&2, better 0): LSGM helps 1&2 but hurts 0 → try two-phase approach
- If v5 < v1: LSGM is responsible for most of the gain → focus on tuning gamma

Key insight: Need 5 samples for reliable estimates. GPU non-determinism with 3 samples
makes comparisons unreliable (v3 showed this clearly).
