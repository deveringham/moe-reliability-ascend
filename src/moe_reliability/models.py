###
# models.py
#
# Helpers for differentiating model families.
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

from .config import PROBE_CHOICES, infer_probe_family

__all__ = ["probe_class", "resolve_probe_family", "moe_dimensions"]


def resolve_probe_family(model_id: str, probe: str = "auto") -> str:
    family = infer_probe_family(model_id) if probe == "auto" else probe
    if family not in PROBE_CHOICES[1:]:
        raise ValueError(f"cannot determine router probe family for {model_id!r}; "
                         f"set model.probe to one of {list(PROBE_CHOICES[1:])}")
    return family

# Gets correct probe class for different model families
def probe_class(family: str):
    from .core import monitoring

    return {
        "deepseek": monitoring.MoEProbeDeepSeek,
        "qwen": monitoring.MoEProbeQwen,
        "mistral": monitoring.MoEProbeMistral,
    }[family]


# Gets n_experts, n_routers (layers), k
# Gets empty model so that weights need not be loaded to check params
def moe_dimensions(model_id: str, family: str) -> tuple[int, int, int]:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from .core.hf_models import hf_token

    config = AutoConfig.from_pretrained(model_id, token=hf_token)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    probe = probe_class(family)(model)
    return probe.n_experts, probe.n_routers, probe.k
