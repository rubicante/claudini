# Claudini Reproduction & Extension Plan

## Goal

Reproduce and extend the results from the Claudini paper (arXiv:2603.24511) — an autoresearch pipeline that uses Claude Code to discover adversarial attack algorithms against LLM safety mechanisms. Start with baseline evaluations on small models, then work toward running the autoresearch loop on a budget.

## Compute Setup

- **Local**: MacOS (Apple Silicon). Claude Code runs here. No GPU.
- **Remote**: RunPod community cloud, RTX 4090 (24GB VRAM), SSH access.
- GPU jobs run on RunPod; analysis and code iteration happen locally.

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

YAML presets in `configs/` specify: model, dtype, optim_length, max_flops, samples, seeds, input_spec (source type, layout, init). CLI flags override any preset value.

Results: `results/<method>/<preset_name>/<model>/sample_<S>_seed_<N>.json`

### Registration

Methods auto-register via `__init_subclass__` when `method_name` is set as a class variable. No manual registry edits needed.

### Existing Baselines (28 methods)

gcg, i_gcg, acg, adc, arca, attn_gcg, autoprompt, beast, bon, cold_attack, degcg, egd, esa, faster_gcg, gbda, gcg_pp, lls, mac, magic, mask_gcg, mc_gcg, pez, pgd, probe_sampling, prs, rails, reg_relax, reinforce, slot_gcg, sm_gcg, tao, tgcg, uat

### Quantization

Already supported! `BenchmarkConfig.load_in_4bit = True` triggers NF4 via bitsandbytes. Just add `load_in_4bit: true` to a config YAML.

### Key Models in Configs

- `Qwen/Qwen2.5-7B-Instruct` — main model for random/safeguard tracks (7B, needs ~14GB bf16)
- `openai-community/gpt2` — demo track (124M, CPU-friendly)
- `openai/gpt-oss-safeguard-20b` — safeguard track (20B, needs quantization or multi-GPU)

## Phased Plan

### Phase 1: Orientation (Local, no GPU)

- [x] Read all method implementations and understand the codebase
- [x] Document how TokenOptimizer, FLOP counting, and configs work (this file)
- [ ] Read a few Claude-designed methods (e.g. v1, v50, v100) to understand what the autoresearch loop produces
- [ ] Study the autoresearch skill prompt (`.claude/skills/claudini/SKILL.md`)

**Cost: $0 (Claude Code session only)**

### Phase 2: Smoke Tests (RunPod 4090)

- [ ] Set up RunPod SSH and sync repo
- [ ] Run GCG on demo_train (GPT-2, 1e15 FLOPs) to verify pipeline: `uv run -m claudini.run_bench demo_train --method gcg --sample 0 --results-dir /tmp/smoke`
- [ ] Run GCG on random_train with tight budget: `uv run -m claudini.run_bench random_train --method gcg --sample 0 --max-flops 1e13 --results-dir /tmp/smoke`
- [ ] Verify result JSON structure and loss curves

**Cost: <$1 GPU time**

### Phase 3: Baseline Reproduction (RunPod 4090)

- [ ] Run all baselines on random_train (5 samples, 1e15 FLOPs): `uv run -m claudini.run_bench random_train --max-flops 1e15`
- [ ] Download paper's precomputed results from GitHub releases
- [ ] Compare: method ranking, loss distributions, convergence behavior
- [ ] Identify divergences and document

**Cost: ~$5-10 GPU time (estimate: 5-10 hours @ 1e15 FLOPs per method)**

### Phase 4: Quantization Experiment

- [ ] Run same baselines on `random_train_q4` config (4-bit NF4 quantized Qwen2.5-7B)
- [ ] Compare attack success and loss against Phase 3 full-precision results
- [ ] Evaluate: Does quantized surrogate preserve gradient quality for attack discovery?
- [ ] Document findings

**Cost: ~$3-5 GPU time (quantized model is faster)**

### Phase 5: Autoresearch Loop

- [ ] Run `/claudini` skill for 10-15 iterations targeting Qwen2.5-7B with 1e15 FLOPs
- [ ] Evaluate: Does the agent discover methods that improve on baselines?
- [ ] Analyze the agent log and method evolution chain
- [ ] Compare against paper's Claude-designed methods

**Cost: ~$10-20 GPU time + ~$5-15 Claude API**
