"""
Claude my_run v3: gamma=0.85 (proven best) + small noise kicks on patience.

Analysis of v2 (mean=8.85 vs v1's 8.53):
- v2 had two changes: lower gamma (0.75) + noisy kick (scale=1.5)
- Sample 1 regressed: v2=8.40 vs v1=7.10
- v2 sample 1 converged to 8.40 at step ~300, never getting v1's late breakthrough at step ~400
- Likely cause: gamma=0.75 changed gradient dynamics, disrupting the convergence path

v3 strategy: isolate the noise kick without gamma change.
- Keep gamma=0.85 (proven optimal in v1)
- Add noise kicks on every patience trigger (not alternating), noise_scale=0.5 (gentle)
- Gentle noise = maintain good directional information while exploring nearby region
- No alternating: every reset uses noisy kick since v1 already showed that clean
  one-hot resets are insufficient to escape hard basins (sample 0 stuck 400 steps)
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from claudini.methods.claude_my_run.v1 import ClaudeMyRunV1Optimizer

logger = logging.getLogger("claudini")


class ClaudeMyRunV3Optimizer(ClaudeMyRunV1Optimizer):
    """gamma=0.85 + gentle noisy patience escape (scale=0.5)."""

    method_name = "claude_my_run_v3"

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
        noise_scale: float = 0.5,
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
        self.noise_scale = noise_scale

    def setup(self, prompt: str, target: str) -> None:
        super().setup(prompt, target)
        logger.info(
            "claude_my_run_v3: LSGM(gamma=%.2f) + gentle noisy patience escape(scale=%.2f)",
            self.lsgm_gamma,
            self.noise_scale,
        )

    def _patience_check(
        self,
        step: int,
        discrete_loss: float,
        relaxed_loss: float,
        embedding_factors: Tensor,
    ) -> None:
        """Every patience trigger: noisy kick from best factors.

        Adds small Gaussian noise to best_embedding_factors before projecting.
        Soft enough to preserve directional information while escaping basins.
        """
        improved = False
        if discrete_loss < self.best_discrete_loss:
            self.best_discrete_loss = discrete_loss
            improved = True
        if relaxed_loss < self.best_relaxed_loss:
            self.best_relaxed_loss = relaxed_loss
            improved = True

        if improved:
            self.best_embedding_factors = embedding_factors.detach().clone()
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        if self.steps_without_improvement >= self.patience_limit:
            # Noisy kick: perturb best factors gently and project to simplex
            noise = torch.randn_like(self.best_embedding_factors) * self.noise_scale
            noisy = self.best_embedding_factors + noise
            if self.forbidden_mask is not None:
                noisy[..., self.forbidden_mask] = -1e9
            projected = F.softmax(noisy.view(-1, noisy.shape[-1]), dim=-1).view_as(noisy)
            embedding_factors.data.copy_(projected)
            self.steps_without_improvement = 0
            self.log("noise_kick", 1, prog_bar=False)
