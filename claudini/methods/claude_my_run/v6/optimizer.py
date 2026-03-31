"""
Claude my_run v6: LSGM(0.85) + longer cosine LR cycles (T_0=120, T_mult=2).

Analysis of v1 hard samples (0, 3, 4):
- Hard samples plateau at 10-13 range after 100-300 steps
- v1 uses PGD's hardcoded T_0=60 cosine cycles (8 cycles in 490 active steps)
- Each cycle: LR oscillates 0.40→0.11→0.40 in 60 steps → too short for deep escape
- LR peaks every 60 steps but hard samples stay stuck for 200-400 steps

v6 change: override setup() to use T_0=120, T_mult=2:
- Cycle 1: 120 steps (LR 0.40→0.11→0.40)
- Cycle 2: 240 steps (2x longer)
- Total active steps: 490 → 1 full + partial of 2nd cycle

The longer cycle hypothesis: hard samples need more sustained high-LR exploration
before they can escape. With T_0=120, the optimizer spends 60 consecutive steps near
LR=0.40 (vs only 30 with T_0=60), giving more opportunity for large gradient steps.

All other v1 settings preserved: LSGM(0.85), stripped aux losses.
"""

import logging

from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingWarmRestarts, SequentialLR
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from claudini.methods.claude_my_run.v1 import ClaudeMyRunV1Optimizer

logger = logging.getLogger("claudini")


class ClaudeMyRunV6Optimizer(ClaudeMyRunV1Optimizer):
    """LSGM(0.85) + stripped losses + longer cosine LR cycles."""

    method_name = "claude_my_run_v6"

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
        cosine_t0: int = 120,
        cosine_t_mult: int = 2,
        seed: int | None = None,
        allow_non_ascii: bool = False,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optim_length=optim_length,
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
            lsgm_gamma=lsgm_gamma,
            seed=seed,
            allow_non_ascii=allow_non_ascii,
        )
        self.cosine_t0 = cosine_t0
        self.cosine_t_mult = cosine_t_mult

    def setup(self, prompt: str, target: str) -> None:
        # Call v1 setup (which registers LSGM hooks + calls PGD.setup)
        super().setup(prompt, target)

        # Override the LR scheduler with longer cycles
        sched1 = ConstantLR(self.optimizer, factor=1.0, total_iters=100)
        sched2 = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.cosine_t0,
            T_mult=self.cosine_t_mult,
            eta_min=self.lr_max,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[sched1, sched2],
            milestones=[100],
        )

        logger.info(
            "claude_my_run_v6: LSGM(gamma=%.2f) + T_0=%d, T_mult=%d",
            self.lsgm_gamma,
            self.cosine_t0,
            self.cosine_t_mult,
        )
