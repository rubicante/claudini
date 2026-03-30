# Claudini Reproduction & Extension Plan

## Goal

Reproduce and extend the results from the Claudini paper (arXiv:2603.24511) — an autoresearch pipeline that uses Claude Code to discover adversarial attack algorithms against LLM safety mechanisms. Start with baseline evaluations on small models, then work toward running the autoresearch loop on a budget.

## Compute Setup

- **Local**: MacOS M4 MacBook Air. Claude Code runs here. MPS backend works but is slow (~90s per GPT-2 run); not practical for real experiments.
- **Remote**: RunPod community cloud, RTX 4090 (24GB VRAM), SSH access configured (`ssh runpod`).
- GPU jobs run on RunPod; analysis and code iteration happen locally.
- Repo hosted at `rubicante/claudini` (public). Remotes: `origin` = our fork, `upstream` = `romovpa/claudini`.

## Codebase Cheat Sheet

### TokenOptimizer Interface

The core abstraction (`claudini/base.py`). Every attack method subclasses it:

```
setup(prompt, target)    -> tokenize, embed, prepare state
step(step_num)           -> (discrete_loss, soft_loss|None, suffix_string)
```

The `run()` loop calls `setup()` once, then `step()` until FLOP budget or step limit.

### FLOP Counting

Kaplan et al. (2020) formula. Methods must call explicitly:
- `self.flop_counter.count_forward(n_tokens)` -> 2 * N_params * n_tokens
- `self.flop_counter.count_backward(n_tokens)` -> 4 * N_params * n_tokens
- `self.flop_counter.count_forward_backward(n_tokens)` -> 6 * N_params * n_tokens

N_params excludes embeddings. MoE models use active params only.

### Config -> Experiment Mapping

YAML presets in `configs/` specify: model, dtype, optim_length, max_flops, samples, seeds, input_spec (source type, layout, init). CLI flags override any preset value. The `load_in_4bit: true` field enables NF4 quantization via bitsandbytes.

Results: `results/<method>/<preset_name>/<model>/sample_<S>_seed_<N>.json` (gitignored — keep locally).

### Registration

Methods auto-register via `__init_subclass__` when `method_name` is set as a class variable. No manual registry edits needed.

### Key Models in Configs

- `Qwen/Qwen2.5-7B-Instruct` — main model for random/safeguard tracks (7B, ~14GB bf16, ~4GB NF4)
- `openai-community/gpt2` — demo track (124M, CPU-friendly)
- `openai/gpt-oss-safeguard-20b` — safeguard track (20B, needs quantization or multi-GPU)

## Progress Log

### Session 1 (2026-03-27)

**Phase 1 — Orientation.** Read full codebase: base.py (TokenOptimizer, FlopCounter, RunResult), all configs, GCG optimizer, autoresearch skill prompt. Created project scaffolding: PROJECT_PLAN.md, RUNPOD_SETUP.md, COST_LOG.md, baseline_analysis.py.

**Phase 2 — Smoke tests.** Set up RunPod 4090 pod, installed uv and dependencies, cloned repo. Ran GCG on Qwen2.5-7B at 1e13 FLOPs (too small — only 1 step) then 1e15 FLOPs (5 steps, loss 16.0 → 14.6). Pipeline verified end-to-end.

**Phase 3/4 — Baseline sweep (merged).** Attempted full 33-method sweep on bf16 Qwen2.5-7B but hit OOM on memory-hungry methods (acg, etc.) — 24GB insufficient for bf16 + 512-candidate batches. Switched to 4-bit NF4 quantized config (`random_train_q4`). Discovered `load_in_4bit` wasn't being read from YAML configs — fixed in `run_bench.py`. Completed 32/33 baselines on NF4 at 1e15 FLOPs (probe_sampling crashed with CUDA assert; pgd crashed but results were saved). All 160 result JSONs pulled locally.

**Leaderboard (NF4 4-bit, 1e15 FLOPs, top 10):**

| Rank | Method | Mean Loss |
|------|--------|-----------|
| 1 | pgd | 11.33 |
| 2 | gbda | 12.49 |
| 3 | uat | 12.78 |
| 4 | lls | 13.04 |
| 5 | mc_gcg | 13.16 |
| 6 | tao | 13.18 |
| 7 | mac | 13.24 |
| 8 | attngcg | 13.34 |
| 9 | tgcg | 13.36 |
| 10 | arca | 13.65 |

**Paper comparison.** Downloaded 14,497 precomputed results from paper's GitHub release. Compared baseline rankings: Spearman ρ = 0.495 (p = 0.005) between paper (bf16, 1e17 FLOPs) and our results (NF4, 1e15 FLOPs). Correlation is significant but moderate — divergence is mostly budget-driven (continuous methods like pgd/gbda shine at low budgets; discrete methods like i_gcg/gcg need more steps). Bottom of rankings matches well.

**Other fixes.** Added `from __future__ import annotations` to base.py and input_spec.py for Python 3.13 compat (upstream targets 3.14 which has PEP 649). Verified MPS backend works on M4 MacBook Air but too slow for practical use (~45min per GPT-2 run at 1e15; ~90s with optimized config at 1e14).

## Phased Plan

### Phase 1: Orientation (Local, no GPU) — DONE

- [x] Read all method implementations and understand the codebase
- [x] Document how TokenOptimizer, FLOP counting, and configs work (this file)
- [ ] Read a few Claude-designed methods (e.g. v1, v50, v100) to understand what the autoresearch loop produces
- [ ] Study the autoresearch skill prompt in depth

### Phase 2: Smoke Tests (RunPod 4090) — DONE

- [x] Set up RunPod SSH and sync repo
- [x] Run GCG on Qwen2.5-7B with tight budget (1e13, 1e15 FLOPs)
- [x] Verify result JSON structure and loss curves

### Phase 3: Baseline Reproduction — PARTIALLY DONE

- [x] Run baselines on random_train_q4 (32/33 methods, 5 samples, 1e15 FLOPs)
- [x] Download paper's precomputed results (14,497 runs)
- [x] Compare rankings (Spearman ρ = 0.495, p = 0.005)
- [ ] Run bf16 baselines (blocked by OOM — would need reduced num_candidates or larger GPU)

### Phase 4: Quantization Experiment — DONE (merged with Phase 3)

- [x] Run baselines on `random_train_q4` config (NF4 Qwen2.5-7B)
- [x] Compare against paper's bf16 results
- [x] Conclusion: NF4 is viable — rankings correlate, gradients work, fits in 24GB VRAM

### Phase 5: Autoresearch Loop — NOT STARTED

- [ ] Complete one-time RunPod pod setup (see `docs/RUNPOD_SETUP.md`)
- [ ] Smoke-test the pipeline end-to-end (one job, `--max-flops 1e12`, confirm issue closes and results appear)
- [ ] Run `/claudini` skill for 10-15 iterations targeting Qwen2.5-7B with 1e15 FLOPs
- [ ] Evaluate whether the agent discovers methods that improve on baselines
- [ ] Analyze the agent log and method evolution chain

**Benchmark pipeline** (implemented in session 2, 2026-03-30): jobs submitted as GitHub Issues from local machine, worker daemon on pod processes queue and self-stops, results committed to git. See `claudini/pipeline/` and `claudini/backends/`.

**Open questions for Phase 5:**
- Whether to use NF4 or bf16 with reduced candidates for the autoresearch loop
- Whether 1e15 FLOPs is sufficient budget for the loop to find meaningful improvements

## Files Created

| File | Purpose |
|------|---------|
| `docs/PROJECT_PLAN.md` | This file |
| `docs/RUNPOD_SETUP.md` | One-time pod setup guide (env vars, startup command, bootstrap) |
| `docs/COST_LOG.md` | Manual cost tracking template (GPU + API) |
| `configs/random_train_q4.yaml` | 4-bit NF4 quantized config for Qwen2.5-7B |
| `configs/demo_train_fast.yaml` | Low-budget GPT-2 config for quick local iteration |
| `notebooks/baseline_analysis.py` | Results loader + leaderboard generator |

## Code Changes to Upstream

| File | Change |
|------|--------|
| `claudini/run_bench.py` | Pass `load_in_4bit` from YAML config to BenchmarkConfig |
| `claudini/base.py` | Add `from __future__ import annotations` (Python 3.13 compat) |
| `claudini/input_spec.py` | Add `from __future__ import annotations` (Python 3.13 compat) |
| `pyproject.toml` | Add `bitsandbytes` optional dependency under `[project.optional-dependencies]` |
