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
from .engines.offload import (  # noqa: E402
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
from .engines.fast import (  # noqa: E402
    disable_fast,
    disable_fast_train,
    enable_fast,
    enable_fast_train,
    fast_available,
)
from .engines.batched import (  # noqa: E402
    batched_train_available,
    disable_batched_train,
    enable_batched_train,
)
from .engines.cold_engine import cold_engine_available, disable_cold_engine, enable_cold_engine  # noqa: E402
from .engines.placement import (  # noqa: E402
    force_cold_mass,
    load_manifest,
    save_manifest,
    solve_placement,
    verify_manifest,
)
from .engines.hybrid import (  # noqa: E402
    disable_hybrid_tier,
    enable_hybrid_tier,
    hybrid_available,
    cold_stats,
    prefetch_stats,
    set_prefetch,
)
from .engines.cpu_router import (  # noqa: E402
    cpu_router_available,
    disable_cpu_router,
    enable_cpu_router,
    router_trip_stats,
)
from .engines.hot_residency import (  # noqa: E402
    disable_hot_residency,
    dispatched_modules,
    enable_hot_residency,
    hot_residency_available,
)
from .engines.pipelined import disable_pipelined_residency, enable_pipelined_residency, pipelined_available  # noqa: E402
# 0.7.0 surfaces: serving a model whose DENSE side, not just its experts, exceeds
# what the host can hold. Imported lazily-tolerant -- dense_disk needs nothing
# exotic, but keeping the top-level import total means a broken optional dep
# cannot make `import experts4bit_qlora` fail.
from .engines.dense_offload import dense_offload_report, enable_dense_offload  # noqa: E402
from .formats.dense_disk import DenseDiskSource, DiskHome, disk_homes_for  # noqa: E402
from .engines.nvme_experts import (  # noqa: E402
    build_meta_experts,
    disable_mxfp4_nvme_residency,
    enable_mxfp4_nvme_residency,
    enable_nvme_residency,
    expert_geometry_from_arena,
)
# The training counterpart of the above: `enable_nvme_residency` serves frozen
# experts off an arena, this one TRAINS an adapter over them. Kept in its own
# module because they are opposites at the seam -- one replaces the module's
# forward, the other exists to leave it alone.
from .engines.nvme_train import (  # noqa: E402
    arena_train_stats,
    enable_nvme_train_residency,
)
# 0.8.0: the hot-set dial. The README tells you to rank a routing histogram rather than
# take experts by index (+37.1% at identical VRAM on V4-Flash), so the two functions that
# do it have to be importable from the top level -- 0.7.1 shipped precisely to fix the
# class of bug where the front page names a symbol that raises ImportError.
from .engines.expert_profile import coverage_from_profile, hot_sets_from_profile  # noqa: E402
# 0.16.2: CUDA-graph decode capture. Exported at the top level for the same reason
# 0.7.1 was cut -- a symbol users cannot import is a symbol that shipped to nobody.
# capture.py imports only torch eagerly (transformers is imported inside the
# functions), so this cannot make `import experts4bit_qlora` fail.
from .engines.capture import CapturedDecoder, capture_decode, probe_capture  # noqa: E402
from .engines.kv_cache import NF4KVCache, kv_nf4_available  # noqa: E402
from .verify import verify_moe_4bit  # noqa: E402

__all__ = [
    "Experts4bit",
    "ExpertsNbit",
    "ExpertsLoRA",
    "LoRALinear",
    "add_attention_lora",
    "cold_engine_available",
    "cpu_router_available",
    "disable_cold_engine",
    "disable_cpu_router",
    "disable_fast",
    "disable_hot_residency",
    "disable_pipelined_residency",
    "enable_cold_engine",
    "enable_cpu_router",
    "enable_expert_offload",
    "enable_hybrid_tier",
    "disable_hybrid_tier",
    "hybrid_available",
    "set_prefetch",
    "cold_stats",
    "prefetch_stats",
    "force_cold_mass",
    "solve_placement",
    "save_manifest",
    "load_manifest",
    "verify_manifest",
    "router_trip_stats",
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
    "enable_nvme_train_residency",
    "arena_train_stats",
    "build_meta_experts",
    "expert_geometry_from_arena",
    "dispatched_modules",
    "hot_sets_from_profile",
    "capture_decode",
    "CapturedDecoder",
    "probe_capture",
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


__version__ = "0.22.0"

from .engines.speculative import speculative_greedy_decode  # noqa: E402,F401

# --- import-path compatibility for the 0.16.x layout -------------------------
# The modules below moved into arch/ formats/ engines/ subpackages. The public API
# is `__all__` above and did NOT move, so top-level imports are unaffected — but a
# published package means somebody may already import a submodule by path, and
# breaking that silently at import time is the worst way to find out. These alias
# the old dotted names to the new modules so `import experts4bit_qlora.awq` and
# `from experts4bit_qlora.awq import X` both keep working.
import sys as _sys  # noqa: E402
from importlib import import_module as _import_module  # noqa: E402

_MOVED = {
    "awq": "formats.awq",
    "axk1": "arch.axk1",
    "batched": "engines.batched",
    "capture": "engines.capture",
    "cold_engine": "engines.cold_engine",
    "compressed_int": "formats.compressed_int",
    "deepseek_v4": "arch.deepseek_v4",
    "dense_disk": "formats.dense_disk",
    "dense_offload": "engines.dense_offload",
    "expert_profile": "engines.expert_profile",
    "fast": "engines.fast",
    "fp8_blocks": "formats.fp8_blocks",
    "glimmer": "arch.glimmer",
    "glimmer_draft": "arch.glimmer_draft",
    "glimmer_load": "arch.glimmer_load",
    "glm5": "arch.glm5",
    "gptoss": "arch.gptoss",
    "gptq": "formats.gptq",
    "hot_residency": "engines.hot_residency",
    "kv_cache": "engines.kv_cache",
    "mixtral": "arch.mixtral",
    "moe_conventions": "arch.moe_conventions",
    "moe_load": "arch.moe_load",
    "moe_plan": "arch.moe_plan",
    "mxfp4": "formats.mxfp4",
    "nvfp4": "formats.nvfp4",
    "nvme_experts": "engines.nvme_experts",
    "offload": "engines.offload",
    "pipelined": "engines.pipelined",
    "speculative": "engines.speculative",
}

def _install_legacy_module_aliases():
    _self = _sys.modules[__name__]
    for _old, _new in _MOVED.items():
        try:
            _mod = _import_module(f"{__name__}.{_new}")
        except Exception:            # an optional dep missing must not break the package
            continue
        # BOTH bindings are required, and only one of them is obvious.
        # `sys.modules` alone satisfies `import experts4bit_qlora.awq` -- the import
        # machinery finds the cache entry and stops. But a normal import ALSO binds the
        # submodule as an attribute on its parent package, and a cache hit skips that,
        # so `experts4bit_qlora.awq` as an ATTRIBUTE would fall through to this module's
        # `__getattr__` and raise. Testing only via `importlib.import_module` passes
        # straight over that, which is exactly how it was missed the first time.
        _sys.modules.setdefault(f"{__name__}.{_old}", _mod)
        if not hasattr(_self, _old):
            setattr(_self, _old, _mod)

_install_legacy_module_aliases()
