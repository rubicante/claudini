# Agent Log: claude_my_run

**Goal:** Break Qwen2.5-7B on random strings under 1e15 FLOPs
**Config:** `random_train_q4` (Qwen/Qwen2.5-7B-Instruct, 1e15 FLOPs, 4-bit, samples 0-2, seed 0)
**Branch:** `loop/my_run`
**Method prefix:** `claude_my_run_v`

## Baseline

Best method at 1e15 FLOPs (random_train_q4, Qwen2.5-7B): **PGD at 11.33** (mean over 5 samples).

Key observation: PGD gets ~590 gradient steps at 1e15. It's better than GBDA (12.49) and all others.
GCG gets only ~4 steps at this budget — discrete search is too expensive.

## v1 — PGD + LSGM(0.85) + stripped auxiliary losses

**Key ideas:**
1. LSGM backward hooks on all LayerNorm modules (gamma=0.85): smooth gradient flow
2. Strip suffix_control and suffix_nonrepeat losses: focus 100% gradient on target CE
3. Faster entropy annealing (150 steps vs 250): sharpen distributions earlier
4. Higher patience (150 vs 100): fewer disruptive resets

**Issue:** [#15](https://github.com/rubicante/claudini/issues/15)
**Results:** sample 0: 11.76, sample 1: 7.10, sample 2: 6.73 → **mean=8.53**
**vs baseline:** PGD=11.33 → **improvement: +2.80** ✓

Loss curve analysis:
- Sample 0: stuck at 13.37 for steps 100-500, finally reaches 11.76 at step ~500
- Sample 1: big breakthrough jump 10.71→7.10 at step ~400
- Sample 2: late breakthrough 9.77→6.73 near step 590

## v2 — LSGM(0.75) + noisy patience escape (scale=1.5, alternating)

**Issue:** [#16](https://github.com/rubicante/claudini/issues/16)
**Results:** sample 0: 11.51, sample 1: 8.40, sample 2: 6.63 → **mean=8.85**
**vs v1:** Worse overall. Sample 0 improved (+0.25), sample 1 regressed (-1.30).

**Analysis:** Lower gamma (0.75) disrupted sample 1's trajectory. v1 had a big jump to 7.10
at step ~400; v2 converged to 8.40 at step ~300 and stalled. The large noise (1.5) may also
have been too disruptive. Gamma=0.85 is better.

## v3 — LSGM(0.85) + gentle noise kick (scale=0.5, every patience trigger)

**Key ideas:**
1. Restore gamma=0.85 (v1's proven-optimal value)
2. Replace clean one-hot resets with gentle noisy kicks (scale=0.5 vs v2's 1.5)
3. Every patience trigger uses noisy kick (not alternating like v2)

**Hypothesis:** v2's large noise + wrong gamma caused regression. Small noise (0.5) + gamma=0.85
should help sample 0 escape its 13.37 plateau while keeping samples 1&2 on good trajectories.

**Issue:** (pending)
**Best loss:** (pending)

## What to try next (v4)

- If v3 beats v1: try tuning noise_scale (0.3, 0.7) or patience (100, 200)
- If v3 fails: try K=2 restarts with gamma=0.85 (more diversity, fewer steps per restart)
- If variance still high: try larger sample set (samples 0-4) for better signal
