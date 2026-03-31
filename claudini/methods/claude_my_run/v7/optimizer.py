"""
Claude my_run v7: Layer-selective LSGM — apply only to upper half of transformer.

Analysis:
- v1 (LSGM all layers, gamma=0.85): great for s1,s2 but mediocre for s0,s3,s4
- v6 (longer LR cycles): helps s0 but hurts s1,s2
- v5 ablation: NO LSGM is worse than PGD — LSGM is essential

Key question: Does uniform LSGM (all layers) give optimal gradient modification?
Hypothesis: Early layers handle syntax/context; later layers handle semantic token selection.
For token-forcing loss, the later layers (where output logits originate) may be more
important. Applying LSGM only to layers 14-27 (upper half) concentrates the gradient
smoothing where it matters most for token prediction.

Changes vs v1:
- LSGM applied only to layers with index >= num_layers//2 (14 of 28)
- All other settings identical to v1 (gamma=0.85, stripped aux losses)

Secondary hypothesis: uniform LSGM may over-smooth early-layer gradients that carry
important structural signal, while insufficient smoothing in later layers. Layer-selective
LSGM provides a different trade-off.
"""

import logging
import re

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from claudini.methods.claude_my_run.v1 import ClaudeMyRunV1Optimizer

logger = logging.getLogger("claudini")

_LAYER_IDX_RE = re.compile(r"\.layers?\.(\d+)\.")


class ClaudeMyRunV7Optimizer(ClaudeMyRunV1Optimizer):
    """LSGM applied only to the upper half of transformer layers."""

    method_name = "claude_my_run_v7"

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
        layer_fraction: float = 0.5,
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
        self.layer_fraction = layer_fraction

    def _get_norm_modules(self):
        """Return only LayerNorm modules from the upper fraction of transformer layers."""
        # Determine max layer index
        max_layer = -1
        for name, _ in self.model.named_modules():
            m = _LAYER_IDX_RE.search(name)
            if m:
                max_layer = max(max_layer, int(m.group(1)))

        if max_layer < 0:
            # Fallback: return all norms (same as v1)
            return super()._get_norm_modules()

        cutoff = int(max_layer * self.layer_fraction)

        norms = []
        norm_names = {
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        }
        for name, module in self.model.named_modules():
            m = _LAYER_IDX_RE.search(name)
            if m and int(m.group(1)) >= cutoff:
                if any(p in name for p in norm_names):
                    norms.append(module)
        return norms

    def setup(self, prompt: str, target: str) -> None:
        super().setup(prompt, target)
        logger.info(
            "claude_my_run_v7: LSGM(gamma=%.2f, upper %.0f%% of layers, %d hooks)",
            self.lsgm_gamma,
            self.layer_fraction * 100,
            len(self._lsgm_handles),
        )
