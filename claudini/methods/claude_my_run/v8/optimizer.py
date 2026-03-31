"""
Claude my_run v8: v1 + Adam state reset on plateau.

Analysis of hard samples in v1:
- s0: stuck at 13.37 for steps 100-500 (400 steps of stagnation)
- s3: stuck at 13.21 for steps 100-300, then stuck again at 10.56 till end
- s4: stuck at 12.73 for steps 100-300, then stuck at 10.84 till ~step 560

The Adam optimizer accumulates momentum (exp_avg) and second moment (exp_avg_sq).
When stuck in a plateau, exp_avg_sq for plateau positions gets very large (many
similar gradient steps accumulate) → effective Adam LR drops toward zero.

Even when the LR is increased by the cosine schedule, Adam's adaptive scaling
keeps the effective update small for "saturated" positions. This prevents escape.

v8 fix: when best_discrete_loss hasn't improved for adam_reset_patience steps,
reset Adam optimizer state (exp_avg and exp_avg_sq) to zero for all parameters.
This forces Adam to re-estimate moment statistics from raw gradients, potentially
breaking the saturation lock and allowing larger updates.

This is complementary to PGD's patience reset (which resets embedding_factors).
We reset the OPTIMIZER STATE instead, keeping the embedding_factors at their
current values. This means the optimizer continues from the same point but
with fresh adaptive learning rates.

The reset fires independently of PGD's patience reset — the two mechanisms
can work together.
"""

import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from claudini.methods.claude_my_run.v1 import ClaudeMyRunV1Optimizer

logger = logging.getLogger("claudini")


class ClaudeMyRunV8Optimizer(ClaudeMyRunV1Optimizer):
    """v1 + Adam optimizer state reset when discrete loss plateaus."""

    method_name = "claude_my_run_v8"

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
        adam_reset_patience: int = 100,
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
        self.adam_reset_patience = adam_reset_patience
        self._discrete_steps_since_improvement = 0
        self._adam_tracked_best = float("inf")
        self._adam_reset_count = 0

    def setup(self, prompt: str, target: str) -> None:
        super().setup(prompt, target)
        self._discrete_steps_since_improvement = 0
        self._adam_tracked_best = float("inf")
        self._adam_reset_count = 0
        logger.info(
            "claude_my_run_v8: LSGM(gamma=%.2f) + Adam state reset every %d plateau steps",
            self.lsgm_gamma,
            self.adam_reset_patience,
        )

    def step(self, step_num: int) -> tuple[float, float | None, str]:
        result = super().step(step_num)
        discrete_loss = result[0]

        # Track discrete loss improvement independently of PGD's relaxed-loss tracking
        if discrete_loss < self._adam_tracked_best - 0.01:
            self._adam_tracked_best = discrete_loss
            self._discrete_steps_since_improvement = 0
        else:
            self._discrete_steps_since_improvement += 1

        if self._discrete_steps_since_improvement >= self.adam_reset_patience:
            # Reset Adam moment estimates so it follows raw gradients again
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p in self.optimizer.state:
                        state = self.optimizer.state[p]
                        if "exp_avg" in state:
                            state["exp_avg"].zero_()
                        if "exp_avg_sq" in state:
                            state["exp_avg_sq"].zero_()
                        if "step" in state:
                            state["step"] = state["step"].zero_() if isinstance(state["step"], torch.Tensor) else 0

            self._discrete_steps_since_improvement = 0
            self._adam_reset_count += 1
            self.log("adam_reset", self._adam_reset_count, prog_bar=False)

        return result
