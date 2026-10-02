###
# models.py
#
# Helpers for differentiating model families.
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

from .config import PROBE_CHOICES, infer_probe_family

__all__ = ["resolve_probe_family", "moe_dimensions"]

#: Configuration keys holding the MoE dimensions, per model family. Routed experts,
#: experts selected per token, and the attribute naming the layers that are dense
#: rather than MoE (DeepSeek replaces the first few).
_MOE_CONFIG_KEYS = {
    "deepseek": ("n_routed_experts", "num_experts_per_tok", "first_k_dense_replace"),
    "qwen": ("num_experts", "num_experts_per_tok", None),
    "mistral": ("num_local_experts", "num_experts_per_tok", None),
}


def resolve_probe_family(model_id: str, probe: str = "auto") -> str:
    family = infer_probe_family(model_id) if probe == "auto" else probe
    if family not in PROBE_CHOICES[1:]:
        raise ValueError(f"cannot determine router probe family for {model_id!r}; "
                         f"set model.probe to one of {list(PROBE_CHOICES[1:])}")
    return family


def moe_dimensions(model_id: str, family: str) -> tuple[int, int, int]:
    """``(n_experts, n_moe_layers, top_k)``, read from the model configuration.

    Only the configuration is read: instantiating the model to inspect its
    modules is both far more expensive and tied to how a given transformers
    version happens to lay the router out.
    """
    from transformers import AutoConfig

    from .core.hf_models import hf_token

    try:
        experts_key, topk_key, dense_key = _MOE_CONFIG_KEYS[family]
    except KeyError:
        raise ValueError(f"unknown model family {family!r}; expected one of {sorted(_MOE_CONFIG_KEYS)}") from None

    config = AutoConfig.from_pretrained(model_id, token=hf_token)
    missing = [k for k in (experts_key, topk_key) if getattr(config, k, None) is None]
    if missing:
        raise ValueError(f"{model_id} has no {', '.join(missing)} in its configuration; "
                         f"model.probe={family!r} may be wrong for it")

    n_experts = int(getattr(config, experts_key))
    k = int(getattr(config, topk_key))
    n_layers = int(config.num_hidden_layers)
    if dense_key is not None:
        n_layers -= int(getattr(config, dense_key, 0) or 0)
    return n_experts, n_layers, k
