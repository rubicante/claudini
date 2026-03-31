"""
Claude my_run v2: v1 + lower LSGM gamma (0.75) + noisy patience escape.

v1 (8.53 mean) showed sample 0 getting stuck at loss=13.37 for ~400 steps.
Analysis: PGD's patience reset returns to one-hot of best tokens — same basin.
Fix: add Gaussian noise before projecting back, to escape the local minimum.

Changes vs v1:
1. Lower LSGM gamma (0.75 vs 0.85): more aggressive LayerNorm gradient smoothing,
   closer to the 0.68-0.70 found optimal in claude_random at 1e17 FLOPs.
2. Noisy patience reset: instead of resetting to a clean one-hot, add Gaussian
   noise (scale=1.5) to the best embedding factors before projecting to simplex.
   This preserves the "good token direction" while exploring nearby distributions.
3. Alternating strategy: even-numbered resets use noisy kick, odd-numbered use
   clean one-hot (paper's approach) — allows both exploration and exploitation.
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from claudini.methods.claude_my_run.v1 import ClaudeMyRunV1Optimizer

logger = logging.getLogger("claudini")


class ClaudeMyRunV2Optimizer(ClaudeMyRunV1Optimizer):
    """v1 + lower gamma + noisy patience escape to avoid stuck local minima."""

    method_name = "claude_my_run_v2"

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
        lsgm_gamma: float = 0.75,
        noise_scale: float = 1.5,
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
        self._reset_count = 0

    def setup(self, prompt: str, target: str) -> None:
        super().setup(prompt, target)
        self._reset_count = 0
        logger.info(
            "claude_my_run_v2: LSGM(gamma=%.2f) + noisy patience escape(scale=%.2f)",
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
        """Override: alternate between noisy kicks and clean resets.

        Even-numbered resets: add Gaussian noise to best factors → explore nearby region.
        Odd-numbered resets: clean one-hot reset (paper's approach) → exploit best found.
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
            if self._reset_count % 2 == 0:
                # Noisy kick: start from best factors + Gaussian noise, project to simplex
                noise = torch.randn_like(self.best_embedding_factors) * self.noise_scale
                noisy = self.best_embedding_factors + noise
                # Mask forbidden tokens
                if self.forbidden_mask is not None:
                    noisy[..., self.forbidden_mask] = -1e9
                # Project to simplex: softmax gives valid distribution
                projected = F.softmax(noisy.view(-1, noisy.shape[-1]), dim=-1).view_as(noisy)
                embedding_factors.data.copy_(projected)
                self.log("noise_kick", 1, prog_bar=False)
            else:
                # Clean one-hot reset (paper's approach)
                best_ids = self._discretize(self.best_embedding_factors)
                one_hot = F.one_hot(best_ids, self.vocab_size).float()
                embedding_factors.data.copy_(one_hot.unsqueeze(0))

            self._reset_count += 1
            self.steps_without_improvement = 0
