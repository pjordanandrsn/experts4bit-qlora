"""experts4bit-qlora — QLoRA fine-tuning of fused 4-bit Mixture-of-Experts on a single small GPU.

``Experts4bit`` resolves to the upstream bitsandbytes class once it ships in a release
(bitsandbytes#1965); until then it falls back to a vendored copy, so this package works on a
stock ``pip install bitsandbytes`` today. When upstream releases it, bump the bitsandbytes floor
and delete ``experts4bit_qlora/_vendor/`` — no API change for callers.
"""

try:
    # Upstream (bitsandbytes#1965) once released; else the vendored copy on stock bitsandbytes.
    from bitsandbytes.nn import Experts4bit as _upstream_experts4bit
except ImportError:
    _upstream_experts4bit = None

# ExpertsLoRA reaches into Experts4bit internals (from_float / _project / _dequantize_expert /
# _expert_matmul), so prefer the upstream class only while it still exposes that surface — a future
# bitsandbytes whose merged Experts4bit diverged from the vendored copy must fall back to the
# vendored implementation rather than silently break at forward time.
if _upstream_experts4bit is not None and all(
    hasattr(_upstream_experts4bit, _attr)
    for _attr in ("from_float", "_project", "_dequantize_expert", "_expert_matmul")
):
    Experts4bit = _upstream_experts4bit
else:
    from ._vendor.experts import Experts4bit

from .lora import ExpertsLoRA, LoRALinear, add_attention_lora
from .offload import enable_expert_offload, enable_inference_prefetch, offload_model_experts

__all__ = [
    "Experts4bit",
    "ExpertsLoRA",
    "LoRALinear",
    "add_attention_lora",
    "enable_expert_offload",
    "enable_inference_prefetch",
    "offload_model_experts",
]
__version__ = "0.1.2"
