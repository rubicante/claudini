# Agent Log: claude_my_run

**Goal:** Break Qwen2.5-7B on random strings under 1e15 FLOPs
**Config:** `random_train_q4` (Qwen/Qwen2.5-7B-Instruct, 1e15 FLOPs, 4-bit, samples 0-4, seed 0)
**Branch:** `loop/my_run`
**Method prefix:** `claude_my_run_v`

## Baseline

PGD: **mean=11.33** (5 samples). Uniform, all ~11.

## v1 — PGD + LSGM(0.85) + stripped aux losses ★ BEST SO FAR (mean=9.25)

**Issue:** [#15](https://github.com/rubicante/claudini/issues/15) + [#20](https://github.com/rubicante/claudini/issues/20)
s0=11.76, s1=7.10, s2=6.73, s3=10.56, s4=10.11 → **mean=9.25** (+2.08 vs PGD)
Bimodal: s1,s2 break well (6-7); s0,s3,s4 mediocre (10-12).
LSGM is essential (v5 ablation: no LSGM → 12.83).

## v2-v4 — gamma, noise, K=2 restarts (all worse)

v2 (γ=0.75+noise): 8.85*, v3 (γ=0.85+noise, patience never fires): 9.42*,
v4 (K=2): 10.85* — more restarts = fewer steps = no breakthroughs. *3-sample estimate.

## v5 — No LSGM ablation

**Issue:** [#19](https://github.com/rubicante/claudini/issues/19) | mean=12.83 → LSGM is the key ingredient.

## v6 — Longer LR cycles (T_0=120, T_mult=2)

**Issue:** [#21](https://github.com/rubicante/claudini/issues/21) | mean=9.98 (5 samples)
Helps hard samples (s0: 11.76→9.49) but hurts easy ones (s2: 6.73→10.06).

## v7 — Upper-50% layer-selective LSGM

**Issue:** [#22](https://github.com/rubicante/claudini/issues/22) | mean=10.46 (5 samples)
Uniformly worse. Partial LSGM causes shallow early convergence (s1 stuck at 10.28 at step 200).
Full LSGM across all layers is essential for dynamic optimization.

## v8 — v1 + Adam state reset on discrete plateau

**Key insight:** Hard samples plateau 200-400 steps. Adam's exp_avg_sq saturates
for "stuck" positions → effective LR → 0 despite cosine schedule. Resetting Adam
state every 100 plateau steps forces fresh gradient estimation from current position.

**Issue:** (pending)
**Results:** (pending)

## What to try next (v9)

- If v8 > v1: tune adam_reset_patience (50, 150)
- If v8 ≈ v1 or worse: try a fundamentally different approach
  - GCG warm start for first ~4 steps (1e14 FLOPs), then v1 for remaining 9e14
  - Randomized smoothing: evaluate loss at perturbed points for gradient estimation
  - Different optimizer: SGD momentum (v1's base PGD uses Adam)
