"""experts4bit-qlora — QLoRA fine-tuning of fused low-bit Mixture-of-Experts on a single small GPU.

``ExpertsNbit`` (with the ``Experts4bit`` 4-bit subclass) resolves to the upstream bitsandbytes
class once it ships in a release (bitsandbytes#1965); until then it falls back to a vendored copy,
so this package works on a stock ``pip install bitsandbytes`` today. When upstream releases it, bump
the bitsandbytes floor and delete ``experts4bit_qlora/_vendor/`` — no API change for callers.

``ExpertsNbit`` supports nf4/fp4 (4-bit), int8/fp8 (8-bit blockwise), and bf16/fp16 (passthrough)
expert storage; ``Experts4bit`` is the 4-bit-only subclass kept for the original API.
"""

try:
    # Upstream (bitsandbytes#1965) once released; else the vendored copy on stock bitsandbytes.
    from bitsandbytes.nn import Experts4bit as _upstream_experts4bit, ExpertsNbit as _upstream_experts_nbit
except ImportError:
    _upstream_experts4bit = None
    _upstream_experts_nbit = None

# ExpertsLoRA reaches into the base internals (from_float / _project / _dequantize_expert), so
# prefer the upstream classes only while they still expose that surface — a future bitsandbytes
# whose merged classes diverged from the vendored copy must fall back to the vendored implementation
# rather than silently break at forward time. Both names must resolve to the *same* implementation
# (upstream or vendored), never a mix, so ExpertsLoRA's assumptions hold for either base class.
if (
    _upstream_experts4bit is not None
    and _upstream_experts_nbit is not None
    and all(
        hasattr(cls, attr)
        for cls in (_upstream_experts4bit, _upstream_experts_nbit)
        for attr in ("from_float", "_project", "_dequantize_expert")
    )
):
    Experts4bit = _upstream_experts4bit
    ExpertsNbit = _upstream_experts_nbit
else:
    from ._vendor.experts import Experts4bit, ExpertsNbit

# Package-owned regardless of which class implementation is adopted above: the canonical scheme
# names and their accepted aliases are this package's contract, not upstream's.
from ._vendor.experts import normalize_quant_type
from .lora import ExpertsLoRA, LoRALinear, add_attention_lora
from .offload import (
    enable_expert_offload,
    enable_inference_prefetch,
    offload_model_experts,
    offload_stats_report,
    report_offload_environment,
    reset_offload_stats,
)

__all__ = [
    "Experts4bit",
    "ExpertsNbit",
    "ExpertsLoRA",
    "LoRALinear",
    "add_attention_lora",
    "enable_expert_offload",
    "enable_inference_prefetch",
    "normalize_quant_type",
    "offload_model_experts",
    "offload_stats_report",
    "report_offload_environment",
    "reset_offload_stats",
]
__version__ = "0.2.0"
