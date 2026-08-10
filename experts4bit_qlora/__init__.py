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

# ExpertsLoRA reaches into the base internals (from_float / _project / _dequantize_expert), the
# loader's class dispatch assumes Experts4bit IS an ExpertsNbit, and this package promises the
# state_dict metadata contract (get/set_extra_state) — so prefer the upstream classes only while
# they satisfy all of that; a future bitsandbytes whose merged classes diverged from the vendored
# copy must fall back to the vendored implementation rather than silently break at forward or load
# time. Both names must resolve to the *same* implementation (upstream or vendored), never a mix,
# so ExpertsLoRA's assumptions hold for either base class.
def _upstream_contract_ok(experts_4bit, experts_nbit) -> bool:
    from torch import nn as _nn

    return (
        # issubclass(X, X) is True, so an upstream that aliases both names still qualifies.
        issubclass(experts_4bit, experts_nbit)
        and all(
            hasattr(cls, attr)
            for cls in (experts_4bit, experts_nbit)
            for attr in ("from_float", "_project", "_dequantize_expert")
        )
        # get/set_extra_state exist on every nn.Module (as raising stubs) — require real OVERRIDES,
        # i.e. an upstream that actually implements the metadata contract.
        and experts_nbit.get_extra_state is not _nn.Module.get_extra_state
        and experts_nbit.set_extra_state is not _nn.Module.set_extra_state
    )


if (
    _upstream_experts4bit is not None
    and _upstream_experts_nbit is not None
    and _upstream_contract_ok(_upstream_experts4bit, _upstream_experts_nbit)
):
    Experts4bit = _upstream_experts4bit
    ExpertsNbit = _upstream_experts_nbit
else:
    from ._vendor.experts import Experts4bit, ExpertsNbit

# These imports must follow the class resolution above (lora/offload import the resolved names),
# hence the E402s. normalize_quant_type is package-owned regardless of which implementation is
# adopted: the canonical scheme names and their accepted aliases are this package's contract.
from ._vendor.experts import normalize_quant_type  # noqa: E402
from .lora import ExpertsLoRA, LoRALinear, add_attention_lora  # noqa: E402
from .offload import (  # noqa: E402
    enable_decode_stack,
    offload_handles,
    enable_expert_offload,
    enable_inference_prefetch,
    enable_expert_cache,
    enable_routed_staging,
    enable_speculative_staging,
    speculative_stats,
    offload_model_experts,
    offload_stats_report,
    report_offload_environment,
    reset_offload_stats,
)

# verify_moe_4bit only touches the resolved Experts4bit/ExpertsNbit classes (core deps), so it is
# safe to import eagerly. The streaming loader is NOT — see __getattr__ below.
from .fast import (  # noqa: E402
    disable_fast,
    disable_fast_train,
    enable_fast,
    enable_fast_train,
    fast_available,
)
from .batched import (  # noqa: E402
    batched_train_available,
    disable_batched_train,
    enable_batched_train,
)
from .cold_engine import cold_engine_available, disable_cold_engine, enable_cold_engine  # noqa: E402
from .hot_residency import (  # noqa: E402
    disable_hot_residency,
    dispatched_modules,
    enable_hot_residency,
    hot_residency_available,
)
from .pipelined import disable_pipelined_residency, enable_pipelined_residency, pipelined_available  # noqa: E402
# 0.7.0 surfaces: serving a model whose DENSE side, not just its experts, exceeds
# what the host can hold. Imported lazily-tolerant -- dense_disk needs nothing
# exotic, but keeping the top-level import total means a broken optional dep
# cannot make `import experts4bit_qlora` fail.
from .dense_offload import dense_offload_report, enable_dense_offload  # noqa: E402
from .dense_disk import DenseDiskSource, DiskHome, disk_homes_for  # noqa: E402
from .nvme_experts import (  # noqa: E402
    build_meta_experts,
    disable_mxfp4_nvme_residency,
    enable_mxfp4_nvme_residency,
    enable_nvme_residency,
    expert_geometry_from_arena,
)
# 0.8.0: the hot-set dial. The README tells you to rank a routing histogram rather than
# take experts by index (+37.1% at identical VRAM on V4-Flash), so the two functions that
# do it have to be importable from the top level -- 0.7.1 shipped precisely to fix the
# class of bug where the front page names a symbol that raises ImportError.
from .expert_profile import coverage_from_profile, hot_sets_from_profile  # noqa: E402
from .kv_cache import NF4KVCache, kv_nf4_available  # noqa: E402
from .verify import verify_moe_4bit  # noqa: E402

__all__ = [
    "Experts4bit",
    "ExpertsNbit",
    "ExpertsLoRA",
    "LoRALinear",
    "add_attention_lora",
    "cold_engine_available",
    "disable_cold_engine",
    "disable_fast",
    "disable_hot_residency",
    "disable_pipelined_residency",
    "enable_cold_engine",
    "enable_expert_offload",
    "NF4KVCache",
    "kv_nf4_available",
    "enable_fast",
    "enable_hot_residency",
    "enable_pipelined_residency",
    "fast_available",
    "hot_residency_available",
    "pipelined_available",
    "enable_inference_prefetch",
    "enable_expert_cache",
    "enable_decode_stack",
    "offload_handles",
    "enable_routed_staging",
    "enable_speculative_staging",
    "speculative_stats",
    "normalize_quant_type",
    "offload_model_experts",
    "offload_stats_report",
    "report_offload_environment",
    "reset_offload_stats",
    "enable_fast_train",
    "enable_batched_train",
    "disable_batched_train",
    "batched_train_available",
    "disable_fast_train",
    "enable_dense_offload",
    "dense_offload_report",
    "DenseDiskSource",
    "DiskHome",
    "disk_homes_for",
    "enable_nvme_residency",
    "enable_mxfp4_nvme_residency",
    "disable_mxfp4_nvme_residency",
    "build_meta_experts",
    "expert_geometry_from_arena",
    "dispatched_modules",
    "hot_sets_from_profile",
    "coverage_from_profile",
    "verify_moe_4bit",
    # Provided lazily by __getattr__ below (importing them pulls in the [train] extra).
    "load_moe_4bit_streaming",
    "load_olmoe_4bit_streaming",
]


# `load_moe_4bit_streaming` / `load_olmoe_4bit_streaming` live in `.loader`, which imports
# transformers + accelerate + safetensors + huggingface_hub (the `[train]` extra) at module top.
# Exposing them lazily (PEP 562) keeps `import experts4bit_qlora` working on a core-only install —
# you pay that heavy import only when you actually reach for the streaming loader.
_LAZY_LOADER_EXPORTS = ("load_moe_4bit_streaming", "load_olmoe_4bit_streaming")


def __getattr__(name):
    if name in _LAZY_LOADER_EXPORTS:
        from . import loader

        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.13.0"
