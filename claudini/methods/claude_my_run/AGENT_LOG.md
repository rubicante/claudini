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

## v8 — v1 + Adam state reset on discrete plateau (mean=10.22, WORSE)

**Key insight:** Hard samples plateau 200-400 steps. Adam's exp_avg_sq saturates
for "stuck" positions → effective LR → 0 despite cosine schedule. Resetting Adam
state every 100 plateau steps forces fresh gradient estimation from current position.

**Issue:** [#23](https://github.com/rubicante/claudini/issues/23)
s0=12.32, s1=7.10, s2=11.31, s3=11.25, s4=9.11 → **mean=10.22** (−1.0 vs v1)

**Post-mortem:** Adam reset backfires on easy samples. s2 collapsed from 6.73 → 11.31.
The reset destroys accumulated momentum that was enabling deep convergence on samples
that were already progressing well. The reset fires indiscriminately — it can hit during
productive phases, not just true plateaus (discrete loss is noisy and may stagnate briefly
even when the relaxed loss is still improving). The improvement on s4 (10.11 → 9.11) is
not enough to offset the regression on s2.

## What to try next (v9) — session ended, picking up here

- **GCG warm start:** Run GCG for the first ~1e14 FLOPs (~60 steps at this budget),
  then switch to v1 (LSGM). GCG does discrete token swaps and can jump out of basins
  that gradient descent can't escape; v1 then refines the continuous relaxation.
- **Randomized smoothing:** Evaluate loss at multiple perturbed embedding points for
  a smoother gradient estimate — may help hard samples that are in rough loss basins.
- **Different optimizer:** Try SGD with momentum or Adagrad (less adaptive than Adam);
  Adam's adaptive denominator is the saturation mechanism.
