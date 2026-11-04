# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Export a trained adapter directory to the PEFT layout SGLang's MoE LoRA loader reads (docs/ADAPTER_PORTABILITY.md §7).

Reads a ``backend-adapter/1`` or ``/2`` directory through :func:`adapter_dir.verify_adapter_dir` and writes
``adapter_model.safetensors``, ``adapter_config.json`` and ``e4b_export.json`` (the provenance) into a new directory.
The tensors are copied, never recomputed or cast:

* **Experts.** ``<stack>.gate_up_lora_{A,B}`` and ``<stack>.down_lora_{A,B}`` (``lora.ExpertsLoRA``: ``[E, r, hidden]``,
  ``[E, 2*intermediate, r]``, ``[E, r, intermediate]``, ``[E, hidden, r]``) become
  ``base_model.model.<stack>.{gate_up_proj,down_proj}.lora_{A,B}.weight`` with the same 3-D shapes. SGLang v0.5.23
  reads a 3-D gate_up ``lora_A`` whose first dimension is the expert count as per-expert factors
  (``srt/lora/lora_manager.py:764-798``). A first dimension of 1 is its shared-outer form, a different
  parameterization this exporter never writes. For a stacked ``gate_up_proj`` SGLang repeats ``lora_A`` to ``2r`` and
  keeps ``lora_B`` (``srt/lora/lora.py:490-498``), gate rows first, as ``ExpertsLoRA`` trains them (``proj.chunk(2)``).
* **Attention.** ``<proj>.lora_{A,B}`` become ``base_model.model.<proj>.lora_{A,B}.weight``.
* **Scale.** ``lora_alpha`` and ``r`` are the manifest setup's. Training applied ``alpha / r`` to both scopes, and SGLang
  applies ``lora_alpha / r`` to every module, so a ``/2`` manifest whose recorded applied scale differs is refused.

What this does NOT give is exactness. SGLang serves an unquantized base or one of its own quantized formats, never NF4,
and it copies the factors into LoRA buffers of its serving dtype (``srt/lora/mem_pool.py:633-709``). On a served base
equal to training's ``Q_train(W)`` the served function is the trained one up to that rounding and to kernel arithmetic.
On any other base it is bounded-loss (ADAPTER_PORTABILITY §3), and no served parity has been read yet.

Usage: ``python -m experts4bit_qlora.adapter_peft ADAPTER_DIR OUT_DIR``. Any refusal exits 2 and names its reason.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from .adapter_dir import AdapterRefused, verify_adapter_dir

PEFT_WEIGHTS = "adapter_model.safetensors"
PEFT_CONFIG = "adapter_config.json"
PROVENANCE = "e4b_export.json"
#: the PEFT key prefix for a causal LM (``PeftModelForCausalLM.base_model.model``)
PREFIX = "base_model.model."
_EXPERT_STACK = re.compile(r"(?P<stack>.+)\.(?P<proj>gate_up|down)_lora_(?P<ab>[AB])")
#: e4b tensor suffix -> the module name SGLang's loader matches for a FusedMoE (``srt/lora/utils.py:455-458``)
_EXPERT_MODULE = {"gate_up": "gate_up_proj", "down": "down_proj"}


class ExportRefused(ValueError):
    """The adapter cannot be exported to the PEFT layout as trained; ``reason`` says why."""

    def __init__(self, reason: str):
        super().__init__(f"export refused: {reason}")
        self.reason = reason


def _expert_stacks(ad) -> dict:
    """``stack path -> {"gate_up_A": tensor, ...}``, every stack complete and shaped as ``ExpertsLoRA`` trains it."""
    stacks: dict = {}
    for name in ad.expert_tensors:
        m = _EXPERT_STACK.fullmatch(name)
        stacks.setdefault(m["stack"], {})[f"{m['proj']}_{m['ab']}"] = ad.tensors[name]
    r = ad.setup["r"]
    for stack, t in stacks.items():
        missing = {"gate_up_A", "gate_up_B", "down_A", "down_B"} - set(t)
        if missing:
            raise ExportRefused(f"{stack}: expert LoRA incomplete, missing {sorted(missing)}")
        if any(x.dim() != 3 for x in t.values()):
            raise ExportRefused(f"{stack}: expert LoRA tensors must be 3-D [E, ., .]")
        e, r_gu, hidden = t["gate_up_A"].shape
        e_b, gate_up_out, r_gu_b = t["gate_up_B"].shape
        e_d, r_dn, inter = t["down_A"].shape
        e_db, hidden_b, r_dn_b = t["down_B"].shape
        if len({e, e_b, e_d, e_db}) != 1 or len({r, r_gu, r_gu_b, r_dn, r_dn_b}) != 1 or hidden_b != hidden:
            raise ExportRefused(f"{stack}: shapes {[list(x.shape) for x in t.values()]} are not one [E, r, hidden] stack "
                                f"of rank {r}")
        if gate_up_out != 2 * inter:
            raise ExportRefused(f"{stack}: gate_up out {gate_up_out} != 2 x intermediate {inter}. A non-gated expert "
                                f"stack uses SGLang's single-slot buffer, which this exporter does not map")
        if e == 1:
            raise ExportRefused(f"{stack}: one expert. SGLang reads a first dimension of 1 as shared-outer factors, "
                                f"not this adapter")
    return stacks


def _check_attention_sets(ad) -> None:
    """SGLang stacks q, k and v into ``qkv_proj``. It reads ``v`` whenever ``q`` is present and zero-fills only a missing
    ``k``, and it stacks nothing for a layer without ``q`` (``srt/lora/lora.py:281-306``). So a layer whose q/k/v LoRA
    lacks q or v would crash at load or lose its k/v deltas. Refused instead."""
    layers: dict = {}
    for module in ad.attention:
        parent, _, leaf = module.rpartition(".")
        if leaf in ("q_proj", "k_proj", "v_proj"):
            layers.setdefault(parent, set()).add(leaf)
    for parent, leaves in sorted(layers.items()):
        if not {"q_proj", "v_proj"} <= leaves:
            raise ExportRefused(f"{parent}: attention LoRA on {sorted(leaves)}; SGLang stacks q, k and v and needs q and "
                                f"v together (k may be absent)")


def _check_applied_scale(ad) -> None:
    """A ``/2`` manifest records the scale the modules held. SGLang applies one ``lora_alpha / r``, so they must agree."""
    applied = ad.manifest.get("lora_scale_applied")
    if applied is None:
        return
    if not isinstance(applied, dict):
        raise ExportRefused(f"lora_scale_applied {applied!r} is not a {{scope: scale}} record")
    for scope, held in applied.items():
        if held is None:
            continue
        if isinstance(held, bool) or not isinstance(held, (int, float)) or abs(float(held) - ad.scale) > 1e-12:
            raise ExportRefused(f"lora_scale_applied.{scope} {held!r} is not alpha/r {ad.scale!r}; SGLang applies "
                                f"lora_alpha/r to every module")


def export_peft(adapter_dir, out_dir) -> dict:
    """Write the PEFT layout of ``adapter_dir`` into ``out_dir`` (new or empty). Returns the provenance record.

    Raises :class:`adapter_dir.AdapterRefused` (7) for a directory that does not verify, and :class:`ExportRefused` for
    an adapter this layout cannot carry as trained.
    """
    from safetensors.torch import save_file

    ad = verify_adapter_dir(adapter_dir)
    if ad.revision is None:
        raise ExportRefused("the manifest has no resolved_model_revision: the adapter corrects one base commit, and "
                            "without it the export cannot say which (ADAPTER_PORTABILITY §3)")
    if ad.other_tensors:
        raise ExportRefused(f"tensors that are neither attention nor expert LoRA: {list(ad.other_tensors)}")
    stacks = _expert_stacks(ad)
    _check_attention_sets(ad)
    _check_applied_scale(ad)
    if not stacks and not ad.attention:
        raise ExportRefused("the adapter holds no LoRA tensors")
    out = Path(out_dir).expanduser()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ExportRefused(f"{out} exists and is not an empty directory; the export never overwrites")

    tensors, targets = {}, set()
    for stack, t in stacks.items():
        for proj, module in _EXPERT_MODULE.items():
            for ab in "AB":
                tensors[f"{PREFIX}{stack}.{module}.lora_{ab}.weight"] = t[f"{proj}_{ab}"].contiguous()
            targets.add(module)
    for module, ab in ad.attention.items():
        for side, name in ab.items():
            tensors[f"{PREFIX}{module}.lora_{side}.weight"] = ad.tensors[name].contiguous()
        targets.add(module.rsplit(".", 1)[-1])

    model = ad.manifest.get("model")
    base_model = model.get("model") if isinstance(model, dict) and isinstance(model.get("model"), str) else None
    config = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "inference_mode": True,
        "base_model_name_or_path": base_model, "revision": ad.revision,
        "r": ad.setup["r"], "lora_alpha": ad.setup["alpha"], "lora_dropout": 0.0, "bias": "none",
        "use_rslora": False, "use_dora": False, "fan_in_fan_out": False,
        "target_modules": sorted(targets),
    }
    provenance = {
        "exporter": "experts4bit_qlora.adapter_peft", "layout": "peft-lora, experts per-expert 3-D",
        "source_adapter_sha256": ad.adapter_id, "source_manifest_sha256": ad.manifest_sha256,
        "source_schema": ad.manifest.get("schema"), "expert_absmax": ad.manifest.get("expert_absmax"),
        "expert_stacks": len(stacks), "attention_modules": len(ad.attention), "scale": ad.scale,
        "exactness": "on a served base equal to training's Q_train(W), the trained function up to serving-dtype rounding "
                     "of the factors and kernel arithmetic; bounded-loss on any other base; no served parity has been "
                     "read (docs/ADAPTER_PORTABILITY.md section 7)",
    }
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / PEFT_WEIGHTS), metadata={"format": "pt"})
    provenance["peft_weights_sha256"] = hashlib.sha256((out / PEFT_WEIGHTS).read_bytes()).hexdigest()
    (out / PEFT_CONFIG).write_bytes((json.dumps(config, indent=2, sort_keys=True) + "\n").encode())
    (out / PROVENANCE).write_bytes((json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode())
    return provenance


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m experts4bit_qlora.adapter_peft ADAPTER_DIR OUT_DIR", file=sys.stderr)
        return 2
    try:
        record = export_peft(args[0], args[1])
    except (AdapterRefused, ExportRefused) as e:
        print(str(e), file=sys.stderr)
        return 2
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
