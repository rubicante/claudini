"""
Claude my_run v1: PGD + LSGM with stripped auxiliary losses.

Key insight: at 1e15 FLOPs (~590 gradient steps), PGD is the best baseline (11.33).
The goal is to improve gradient signal quality rather than quantity.

Changes vs PGD:
1. LSGM backward hooks on all LayerNorm modules (gamma=0.85): scale gradients
   flowing back through LayerNorms, smoothing the optimization landscape.
2. Stripped auxiliary losses (suffix_control, suffix_nonrepeat): these losses
   split gradient capacity away from the main CE objective. At ~590 steps, every
   gradient bit counts — we want 100% of gradient signal on target CE.
3. Entropy annealing starts faster (150 steps vs 250): sharpen distributions
   earlier so the discrete argmax gives lower loss sooner.
4. Higher patience (150 vs 100): fewer disruptive resets from the patience mechanism,
   allowing the optimizer to explore more before being reset to a checkpoint.
"""

import logging

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from claudini.methods.original.pgd import PGDOptimizer

logger = logging.getLogger("claudini")


class ClaudeMyRunV1Optimizer(PGDOptimizer):
    """PGD + LSGM with focused loss for the 1e15 FLOPs regime."""

    method_name = "claude_my_run_v1"

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
        # Stripped auxiliary losses
        suffix_control_weight: float = 0.0,
        suffix_control_next_weight: float = 0.0,
        suffix_nonrepeat_weight: float = 0.0,
        # Keep light entropy reg to maintain smooth distributions
        entropy_reg_weight: float = 1e-4,
        entropy_reg_p: float = 6.0,
        relaxation_gap_scale_threshold: float = 0.1,
        lsgm_gamma: float = 0.85,
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
        self.lsgm_gamma = lsgm_gamma
        self._lsgm_handles: list = []

    def _get_norm_modules(self):
        """Collect all LayerNorm modules in the model."""
        norms = []
        for name, module in self.model.named_modules():
            if any(
                p in name
                for p in [
                    "input_layernorm",
                    "post_attention_layernorm",
                    "pre_feedforward_layernorm",
                    "post_feedforward_layernorm",
                    ".ln_1",
                    ".ln_2",
                    "layernorm",
                    "layer_norm",
                ]
            ):
                norms.append(module)
        return norms

    def _register_lsgm_hooks(self) -> list:
        """Register backward hooks that scale LayerNorm gradients by gamma."""
        handles = []
        gamma = self.lsgm_gamma
        for module in self._get_norm_modules():

            def hook(m, grad_input, grad_output, _gamma=gamma):
                if grad_input[0] is not None:
                    return (grad_input[0] * _gamma,) + grad_input[1:]
                return grad_input

            handles.append(module.register_full_backward_hook(hook))
        return handles

    def _remove_hooks(self) -> None:
        for h in self._lsgm_handles:
            h.remove()
        self._lsgm_handles.clear()

    def setup(self, prompt: str, target: str) -> None:
        super().setup(prompt, target)
        self._lsgm_handles = self._register_lsgm_hooks()
        logger.info(
            "claude_my_run_v1: LSGM(%d hooks, gamma=%.2f) + stripped aux losses, patience=%d",
            len(self._lsgm_handles),
            self.lsgm_gamma,
            self.patience_limit,
        )

    def run(self, prompt, target, num_steps, max_flops=None, max_time=None, **kwargs):
        try:
            return super().run(prompt, target, num_steps, max_flops=max_flops, max_time=max_time, **kwargs)
        finally:
            self._remove_hooks()
