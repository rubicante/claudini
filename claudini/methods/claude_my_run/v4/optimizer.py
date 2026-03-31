"""
Claude my_run v4: v1 + K=2 restarts for diversity.

Analysis:
- v1 (8.53): LSGM + stripped losses works well but patience NEVER fires —
  relaxed loss always improves, so the patience counter never hits 150.
- Variance comes from random initialization: sample 0 gets a bad init that
  leads to a plateau at ~13.37 / 11.76, while samples 1&2 get good inits.
- v3 was identical to v1 in practice (patience never fired), differences
  were GPU non-determinism — 3 samples insufficient for reliable comparison.

v4 strategy: K=2 parallel restarts to reduce initialization sensitivity.
- Two independent random initializations share the same Adam optimizer
- Each step, both restarts forward through model together (batched)
- The best of the two is tracked as the result
- Cost: 2x FLOPs per step → ~295 steps total (vs 590 for K=1)

Hypothesis: two independent shots at the loss landscape reduces the chance
that both end up in the same bad basin. Sample 0 particularly benefits since
its plateau suggests a bad local basin that a second init might avoid.

Risk: samples 1&2 found their breakthroughs at step ~400-590 in K=1, which
is beyond the 295-step budget for K=2. But with two restarts, even if neither
reaches the very deep optimum, both might reach an intermediate good region.
"""

import logging

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from claudini.methods.original.pgd import PGDOptimizer

from ..v1 import ClaudeMyRunV1Optimizer

logger = logging.getLogger("claudini")


class ClaudeMyRunV4Optimizer(ClaudeMyRunV1Optimizer):
    """v1 (LSGM + stripped losses) with K=2 parallel restarts."""

    method_name = "claude_my_run_v4"

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        optim_length: int = 20,
        lr: float = 0.11,
        lr_max: float = 0.40,
        entropy_factor_max: float = 0.4,
        entropy_anneal_steps: int = 150,
        patience: int = 150,
        gradient_clip: float = 20.0,
        first_last_ratio: float = 1.0,
        target_weight: float = 1.0,
        suffix_control_weight: float = 0.0,
        suffix_control_next_weight: float = 0.0,
        suffix_nonrepeat_weight: float = 0.0,
        entropy_reg_weight: float = 1e-4,
        entropy_reg_p: float = 6.0,
        relaxation_gap_scale_threshold: float = 0.1,
        lsgm_gamma: float = 0.85,
        num_starts: int = 2,
        seed: int | None = None,
        allow_non_ascii: bool = False,
    ):
        # Call PGDOptimizer.__init__ directly (not v1's which hardcodes num_starts=1)
        PGDOptimizer.__init__(
            self,
            model=model,
            tokenizer=tokenizer,
            optim_length=optim_length,
            num_starts=num_starts,
            lr=lr,
            lr_max=lr_max,
            entropy_factor_max=entropy_factor_max,
            entropy_anneal_steps=entropy_anneal_steps,
            patience=patience,
            gradient_clip=gradient_clip,
            first_last_ratio=first_last_ratio,
            target_weight=target_weight,
            suffix_control_weight=suffix_control_weight,
            suffix_control_next_weight=suffix_control_next_weight,
            suffix_nonrepeat_weight=suffix_nonrepeat_weight,
            entropy_reg_weight=entropy_reg_weight,
            entropy_reg_p=entropy_reg_p,
            relaxation_gap_scale_threshold=relaxation_gap_scale_threshold,
            seed=seed,
            allow_non_ascii=allow_non_ascii,
        )
        self.lsgm_gamma = lsgm_gamma
        self._lsgm_handles: list = []

    def setup(self, prompt: str, target: str) -> None:
        # Call PGDOptimizer.setup (not v1's) to avoid double-registration of hooks
        PGDOptimizer.setup(self, prompt, target)
        self._lsgm_handles = self._register_lsgm_hooks()
        logger.info(
            "claude_my_run_v4: K=%d, LSGM(gamma=%.2f), stripped aux losses",
            self.num_starts,
            self.lsgm_gamma,
        )

    def run(self, prompt, target, num_steps, max_flops=None, max_time=None, **kwargs):
        try:
            return PGDOptimizer.run(self, prompt, target, num_steps, max_flops=max_flops, max_time=max_time, **kwargs)
        finally:
            self._remove_hooks()
