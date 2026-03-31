# Agent Log: claude_my_run

**Goal:** Break Qwen2.5-7B on random strings under 1e15 FLOPs
**Config:** `random_train_q4` (Qwen/Qwen2.5-7B-Instruct, 1e15 FLOPs, 4-bit, samples 0-4, seed 0)
**Branch:** `loop/my_run`
**Method prefix:** `claude_my_run_v`

## Baseline

PGD (best prior method at 1e15 FLOPs): **11.33 mean** over 5 samples.
- s0=11.33, s1=11.58, s2=11.00, s3=11.07, s4=11.68 → uniform, all hard for PGD.

## v1 — PGD + LSGM(0.85) + stripped aux losses ★ BEST SO FAR

**Issue:** [#15](https://github.com/rubicante/claudini/issues/15) + [#20](https://github.com/rubicante/claudini/issues/20)
**Results (5 samples):** s0=11.76, s1=7.10, s2=6.73, s3=10.56, s4=10.11 → **mean=9.25**
**vs PGD:** +2.08 improvement.

Bimodal: samples 1,2 break well (6-7); samples 0,3,4 mediocre (10-12).
LSGM is essential (v5 ablation shows stripping LSGM → 12.83, WORSE than PGD).

## v2 — LSGM(0.75) + noisy patience kick (scale=1.5)

**Issue:** [#16](https://github.com/rubicante/claudini/issues/16) | mean=8.85 (3 samples) — worse than v1
gamma=0.75 disrupted convergence for easy samples.

## v3 — LSGM(0.85) + patience kick (scale=0.5)

**Issue:** [#17](https://github.com/rubicante/claudini/issues/17) | mean=9.42 (3 samples)
Patience NEVER fires (relaxed loss always improves). v3 = v1 + GPU non-determinism.

## v4 — K=2 restarts + LSGM(0.85)

**Issue:** [#18](https://github.com/rubicante/claudini/issues/18) | mean=10.85 (3 samples)
K=2 → ~295 steps/restart. Breakthroughs happen at 400-590 steps. Too few steps.

## v5 — Stripped losses only, NO LSGM (ablation)

**Issue:** [#19](https://github.com/rubicante/claudini/issues/19) | mean=12.83 → LSGM is essential.
Without LSGM, stripping aux losses makes things WORSE than PGD.

## v6 — LSGM(0.85) + longer cosine LR cycles (T_0=120, T_mult=2)

**Issue:** [#21](https://github.com/rubicante/claudini/issues/21) | mean=9.98 (5 samples)
s0=9.49 (better), s1=8.80 (worse), s2=10.06 (worse), s3=11.43, s4=10.10
Longer cycles help hard samples but disrupt easy samples' convergence.

## v7 — Layer-selective LSGM (upper 50% of layers only)

**Key ideas:**
1. Apply LSGM only to layers 14-27 (upper half of Qwen2.5-7B's 28 layers)
2. Hypothesis: later layers handle semantic token decisions more directly
3. Concentrating LSGM on later layers may give a different tradeoff between
   exploration (early) and gradient smoothing (late)

**Issue:** (pending)
**Results:** (pending)

## What to try next (v8)

- If v7 improves over v1: tune layer_fraction (0.3, 0.7)
- If v7 < v1: try LSGM only on attention norms (not FFN norms) → different smoothing
- Alternative direction: use higher gamma (0.90) for less aggressive smoothing
- Key insight: need to help samples 0, 3, 4 without hurting 1, 2
