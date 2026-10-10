"""DQ11 instrument identities and integrity; imports frameworks only inside calls."""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
import json
import marshal
from pathlib import Path
import re
import sys
import types

HERE = Path(__file__).resolve().parent
helper_path = HERE / "adapter_path_audit.py"
if not helper_path.is_file():
    helper_path = HERE.parent / "dq1/adapter_path_audit.py"
spec = importlib.util.spec_from_file_location("dq11_adapter_audit", helper_path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
PROJECTION = re.compile(r"(?:^|\.)layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_sha(tensor, *, values=False):
    import torch

    tensor = tensor.detach().cpu().contiguous()
    if values and tensor.is_floating_point():
        tensor = tensor.float()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def adapter_slots(model):
    slots = {}
    modules = {}
    for name, module in model.named_modules():
        match = PROJECTION.search(name)
        if match and hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            key = "layers." + ".".join(match.groups())
            modules[key] = module
            for suffix in ("A", "B"):
                adapters = getattr(module, "lora_" + suffix)
                if set(adapters) != {"default"}:
                    raise ValueError("unexpected active adapter set")
                slots[key + "." + suffix] = adapters["default"].weight
    if len(modules) != 224 or len(slots) != 448:
        raise ValueError("partial or changed seven-projection adapter engagement")
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if trainable != {id(p) for p in slots.values()}:
        raise ValueError("unexpected trainable parameters")
    for module in modules.values():
        if (module.r != {"default": 16} or module.lora_alpha != {"default": 32}
                or module.scaling != {"default": 2.0} or any(module.use_dora.values())):
            raise ValueError("adapter rank/scale/DoRA changed")
    return slots, modules


def install_initial(model, path):
    from safetensors.torch import load_file
    import torch

    initial = load_file(str(path))
    slots, _ = adapter_slots(model)
    if set(initial) != set(slots):
        raise ValueError("canonical initializer keys differ from loaded adapters")
    with torch.no_grad():
        for key, parameter in slots.items():
            source = initial[key]
            if tuple(source.shape) != tuple(parameter.shape) or source.dtype != torch.float32:
                raise ValueError("canonical initializer shape/dtype differs")
            parameter.data = parameter.data.to(dtype=torch.float32)
            parameter.copy_(source)
            if tensor_sha(parameter) != tensor_sha(source):
                raise ValueError("canonical initializer copy differs")
    return {key: tensor_sha(p) for key, p in slots.items()}


def base_census(model):
    """All frozen parameter values plus exact NF4 packed/state bytes, including homes.

    Different floating storage dtypes are recorded; canonical FP32 hashes compare
    represented values without hiding BF16 rounding of a genuine FP32 source.
    """
    homes = {}
    for module in model.modules():
        handle = getattr(module, "_dense_offload", None)
        if handle:
            for child, attr, is_parameter, home in handle.slots:
                if is_parameter:
                    homes[(id(child), attr)] = home
    result = {}
    for name, module in model.named_modules():
        clean = re.sub(r"^(?:base_model\.)?(?:model\.)+", "", name)
        clean = clean.removesuffix(".base_layer")
        for attr, parameter in module.named_parameters(recurse=False):
            if parameter.requires_grad:
                continue
            tensor = homes.get((id(module), attr), parameter)
            key = clean + "." + attr
            if key in result:
                raise ValueError("ambiguous canonical frozen parameter")
            result[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                           "values_sha256": tensor_sha(tensor, values=True)}
            state = getattr(parameter, "quant_state", None)
            if state is not None:
                result[key]["packed_sha256"] = tensor_sha(tensor)
                result[key]["quantization"] = {
                    k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": tensor_sha(v)}
                    if hasattr(v, "dtype") else str(v)
                    for k, v in state.as_dict(packed=True).items()}
    if not result:
        raise ValueError("empty frozen census")
    return result


def value_identity(census):
    return {key: {k: v for k, v in row.items() if k != "dtype"} for key, row in census.items()}


def walk_code(code):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from walk_code(constant)


@functools.lru_cache(maxsize=128)
def compiled_identities(source):
    reference = compile(source, "", "exec", dont_inherit=True, optimize=sys.flags.optimize)
    return {(code.co_qualname, code.co_firstlineno,
             hashlib.sha256(marshal.dumps(audit._code_without_paths(code))).hexdigest())
            for code in walk_code(reference)}


def verify_callable(function, source_hashes):
    """Compare loaded executable code with compilation of an authorized source file.

    Never execute the file to derive the reference. Record primitive closure values
    and recursively check callable closure cells; unknown bound data refuse.
    """
    if isinstance(function, functools.partial):
        if function.args or function.keywords not in ({}, {"inplace": False}):
            raise ValueError("unregistered partial binding")
        return {"partial": function.keywords, "function": verify_callable(function.func, source_hashes)}
    target = function.__func__ if inspect.ismethod(function) else function
    if not inspect.isfunction(target):
        raise ValueError("unknown loaded callable kind")
    path = inspect.getsourcefile(target)
    if not path or file_sha(path) not in source_hashes:
        raise ValueError("loaded callable has unauthorized source bytes")
    code = target.__code__
    digest = hashlib.sha256(marshal.dumps(audit._code_without_paths(code))).hexdigest()
    if (code.co_qualname, code.co_firstlineno, digest) not in compiled_identities(Path(path).read_bytes()):
        raise ValueError("loaded runtime code differs from authorized source")
    closure = []
    for cell in target.__closure__ or ():
        value = cell.cell_contents
        if callable(value):
            closure.append(verify_callable(value, source_hashes))
        elif value is None or isinstance(value, (bool, int, float, str)):
            closure.append(value)
        else:
            raise ValueError("unknown callable closure binding")
    return {"callable": audit.callable_identity(function), "closure": closure}


def path_census(model, arm, source_hashes):
    from peft.tuners.lora.bnb import Linear4bit

    slots, modules = adapter_slots(model)
    if any(str(p.dtype) != "torch.float32" for p in slots.values()):
        raise ValueError("adapter storage differs from canonical FP32")
    result = audit.audit_adapter_paths(model, declared_compute={"autocast": False, "tf32": False,
                                                                "base_compute": "bfloat16", "arm": arm})
    identities = {}
    if arm in ("U", "U0"):
        from unsloth.kernels import fast_lora
        from unsloth.models.llama import original_apply_o, original_apply_qkv

        attention = [module for name, module in model.named_modules() if name.endswith(".self_attn")]
        mlps = [module for name, module in model.named_modules() if name.endswith(".mlp")]
        if len(attention) != 32 or len(mlps) != 32:
            raise ValueError("changed decoder layer count")
        for module in attention:
            qkv = fast_lora.apply_lora_qkv if arm == "U" else original_apply_qkv
            out = fast_lora.apply_lora_o if arm == "U" else original_apply_o
            if module.apply_qkv is not qkv or module.apply_o is not out:
                raise ValueError("unexpected fusion QKV/O binding")
        for module in mlps:
            expected = fast_lora.apply_lora_mlp_swiglu if arm == "U" else module.__class__.forward
            if hasattr(module, "_unsloth_forward") or getattr(module.forward, "__func__", None) is not expected:
                raise ValueError("unexpected tiled or fused MLP binding")
        for name in ("LoRA_QKV", "LoRA_W", "LoRA_MLP"):
            cls = getattr(fast_lora, name)
            for method in ("forward", "backward"):
                identities[name + "." + method] = verify_callable(getattr(cls, method), source_hashes)
    for name, module in model.named_modules():
        if hasattr(module, "apply_qkv") or hasattr(module, "apply_o") or name.endswith(".mlp"):
            for attr in ("forward", "apply_qkv", "apply_o", "_unsloth_forward"):
                function = getattr(module, attr, None)
                if function is not None:
                    identities[name + "." + attr] = verify_callable(function, source_hashes)
    for key, module in modules.items():
        if not isinstance(module, Linear4bit):
            raise ValueError("dense adapter is not the pinned PEFT-on-bnb wrapper")
        identities[key + ".forward"] = verify_callable(module.forward, source_hashes)
    if not result["loaded_paths"] or not identities:
        raise ValueError("empty adapter binding audit")
    return {"helper": result, "verified_callables": identities}


def score(model, tokens):
    import math
    import torch
    import torch.nn.functional as F

    model.eval()
    scored = {}
    with torch.no_grad():
        for name in ("alpaca-heldout", "wikitext-test"):
            total, count = 0.0, 0
            for row in tokens[name]:
                ids = torch.tensor([row], device="cuda", dtype=torch.long)
                logits = model(input_ids=ids, use_cache=False).logits
                if tuple(logits.shape[:2]) != tuple(ids.shape) or logits.shape[-1] != 32000:
                    raise ValueError("full heldout logits missing")
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, 32000), ids[:, 1:].reshape(-1), reduction="sum")
                total += float(loss)
                count += ids.numel() - 1
                del logits, loss
            ppl = math.exp(total / count)
            if not math.isfinite(ppl):
                raise ValueError("nonfinite heldout quality")
            scored[name] = {"nll": total, "targets": count, "ppl": ppl}
    model.train()
    return scored
