# Agent Log: claude_my_run

**Goal:** Break Qwen2.5-7B on random strings under 1e15 FLOPs
**Config:** `random_train_q4` (Qwen/Qwen2.5-7B-Instruct, 1e15 FLOPs, 4-bit, samples 0-2, seed 0)
**Branch:** `loop/my_run`
**Method prefix:** `claude_my_run_v`

## Baseline

Best method at 1e15 FLOPs (random_train_q4, Qwen2.5-7B): **PGD at 11.33** (mean over 5 samples).

Key observation: PGD gets ~590 gradient steps at 1e15. It's better than GBDA (12.49) and all others.
GCG gets only ~4 steps at this budget — discrete search is too expensive.

## v1 — PGD + LSGM + stripped auxiliary losses

**Key ideas:**
1. LSGM backward hooks on all LayerNorm modules (gamma=0.85): smooth gradient flow through deep transformer layers
2. Strip suffix_control and suffix_nonrepeat losses: focus 100% gradient capacity on target CE
3. Faster entropy annealing (150 steps vs 250): sharpen distributions earlier → better discrete loss sooner
4. Higher patience (150 vs 100): fewer disruptive resets

**Hypothesis:** PGD's auxiliary losses (suffix_control=0.007, suffix_nonrepeat=0.01) steal gradient capacity at the 1e15 regime where every step counts. Removing them + LSGM should reduce the discrete loss gap.

**Issue:** (pending)
**Best loss:** (pending)
**vs baseline:** (pending)

## What to try next

- If v1 beats PGD: try lower gamma (0.75, 0.85) and K=2 restarts
- If v1 doesn't beat PGD: try GBDA + LSGM (different continuous parameterization)
- Try coordinate-wise approach: gradient-guided position selection with fewer forward passes
