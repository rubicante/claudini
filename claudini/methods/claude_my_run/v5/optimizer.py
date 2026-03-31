"""
Claude my_run v5: Stripped losses only, NO LSGM — ablation to isolate LSGM effect.

v1 (8.53): LSGM + stripped losses — better on samples 1&2 but WORSE on sample 0 vs PGD.
  - sample 0: PGD=11.33, v1=11.76 — LSGM hurts sample 0!
  - sample 1: PGD=11.58, v1=7.10 — LSGM helps dramatically
  - sample 2: PGD=11.00, v1=6.73 — LSGM helps dramatically

v5 isolates: what if we strip the auxiliary losses (suffix_control, nonrepeat) but
don't apply LSGM? This tests whether:
  a) The gains in samples 1&2 come from stripped losses alone (LSGM not needed)
  b) LSGM is actually hurting sample 0 (PGD+LSGM < PGD for sample 0)

Changes vs v1:
- NO LSGM hooks
- Same stripped auxiliary losses (suffix_control=0, nonrepeat=0)
- Same faster entropy annealing (entropy_anneal_steps=150)
- Same patience=150
- Same lr_max=0.40

If v5 > v1: LSGM is actively hurting sample 0 (net negative).
If v5 ≈ v1 (samples 1&2 similar, sample 0 better): LSGM helps 1&2, hurts 0 → adaptive LSGM needed.
If v5 < v1: LSGM is responsible for the gains in 1&2.
"""

import logging

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from claudini.methods.original.pgd import PGDOptimizer

logger = logging.getLogger("claudini")


class ClaudeMyRunV5Optimizer(PGDOptimizer):
    """Stripped auxiliary losses, no LSGM — ablation of v1."""

    method_name = "claude_my_run_v5"

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
        seed: int | None = None,
        allow_non_ascii: bool = False,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optim_length=optim_length,
            num_starts=1,
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

    def setup(self, prompt: str, target: str) -> None:
        super().setup(prompt, target)
        logger.info("claude_my_run_v5: stripped aux losses, NO LSGM — ablation")
