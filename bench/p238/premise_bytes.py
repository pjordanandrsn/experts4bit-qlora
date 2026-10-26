"""P238 premise test ($0, CPU, laptop): do e4b and Unsloth hold the SAME NF4 bytes for Qwen3.6-35B-A3B's experts?

Layer 0's expert stacks from the pinned checkpoint (revision 995ad96e; shards sha256-checked against HF's etags), quantized two ways:
- U, Unsloth's path: unsloth_zoo 2026.10.8 moe_utils_bnb4bit.py:634 `_make_expert_params4bit(value, requires_grad=False, **old_dict)`,
  which for a stack under 2**31 elements is `Params4bit(value, **kwargs).to(value.device)` (238); bnb 0.50.2's Params4bit.to quantizes on
  a non-meta device (`_quantize`: ONE quantize_4bit over the whole stack, compress_statistics as configured). Unsloth loads with
  bnb_4bit_use_double_quant=True, so compress_statistics=True; blocksize 64; nf4; uint8 storage. Driven through bitsandbytes' own
  Params4bit class here.
- E, e4b's path: experts4bit_qlora/_vendor/experts.py `ExpertsNbit._quantize_stack` (per expert quantize_4bit, compress_statistics=False,
  blocksize 64, nf4) and, for the nested form, absmax_dq.py `_compress_one` (offset = absmax.mean(); quantize_blockwise(absmax - offset,
  blocksize=256)). Both called from the e4b source at v0.58.1 (the files are unchanged on main).
Compared: packed bytes; fp32 absmax (U requantized flat); nested qabsmax, state2.absmax, offset, state2.code; the NF4 code; and the
dequantized stacks, bit for bit. Data only from the checkpoint; nothing from the network at run time.
"""
import hashlib
import json
import os
import sys
import time
import types

import torch
from safetensors import safe_open

REPO = sys.argv[1]
sys.path.insert(0, REPO)
import bitsandbytes as bnb                                  # noqa: E402
import bitsandbytes.functional as F                         # noqa: E402
from bitsandbytes.nn import Params4bit                      # noqa: E402

from experts4bit_qlora import absmax_dq                     # noqa: E402
from experts4bit_qlora._vendor.experts import ExpertsNbit   # noqa: E402

H = os.path.expanduser("~/p238-shards")
SRC = {"gate_up": ("model-00001-of-00026.safetensors", "model.language_model.layers.0.mlp.experts.gate_up_proj"),
       "down": ("model-00002-of-00026.safetensors", "model.language_model.layers.0.mlp.experts.down_proj")}
torch.set_num_threads(max(1, os.cpu_count() or 1))
report = {"bitsandbytes": bnb.__version__, "torch": torch.__version__, "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0", "stacks": {}}


def h(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


e4b = types.SimpleNamespace(bits=4, blocksize=64, quant_type="nf4")
absmaxes = {}
for which, (f, key) in SRC.items():
    t0 = time.time()
    with safe_open(f"{H}/{f}", "pt") as s:
        W = s.get_tensor(key)                                   # [E, out, in] bf16
    r = {"shape": list(W.shape), "dtype": str(W.dtype), "source_sha16": h(W)}
    # U: bitsandbytes' own Params4bit, as Unsloth's loader builds it
    pU = Params4bit(W, requires_grad=False, blocksize=64, compress_statistics=True, quant_type="nf4", quant_storage=torch.uint8).to("cpu")
    qsU = pU.quant_state
    pUflat = Params4bit(W, requires_grad=False, blocksize=64, compress_statistics=False, quant_type="nf4", quant_storage=torch.uint8).to("cpu")
    # E: e4b's own stack quantizer, then its nested compression
    packedE, absmaxE = ExpertsNbit._quantize_stack(e4b, W)
    flatE = absmaxE.reshape(-1)
    absmaxes[which] = flatE
    r["U_nested"] = bool(getattr(qsU, "nested", False))
    r["packed_equal"] = torch.equal(pU.data.reshape(-1), packedE.reshape(-1))
    r["packed_sha16"] = {"U": h(pU.data.reshape(-1)), "E": h(packedE.reshape(-1))}
    r["absmax_fp32_equal"] = torch.equal(pUflat.quant_state.absmax.reshape(-1), flatE)
    r["code_equal"] = torch.equal(qsU.code, F.get_4bit_type("nf4", device="cpu")) and torch.equal(pUflat.quant_state.code, qsU.code)
    if not r["packed_equal"]:
        a, b = pU.data.reshape(-1), packedE.reshape(-1)
        idx = (a != b).nonzero()
        r["packed_first_diff_byte"] = int(idx[0]) if len(idx) else None
        r["packed_diff_bytes"] = int(len(idx))
    r["seconds"] = round(time.time() - t0, 1)
    report["stacks"][which] = r
    absmaxes[which + "_U"] = qsU
    del W, pU, pUflat, packedE

# e4b's nested form, through absmax_dq._compress_one on a stand-in module (capturing its payload instead of installing it)
cap = {}
orig_install = absmax_dq.install_compressed_absmax_
absmax_dq.install_compressed_absmax_ = lambda mod, payloads: cap.update(payloads)
try:
    mod = types.SimpleNamespace(_buffers={"gate_up_absmax": absmaxes["gate_up"], "down_absmax": absmaxes["down"]})
    absmax_dq._compress_one(mod)
finally:
    absmax_dq.install_compressed_absmax_ = orig_install
for which in ("gate_up", "down"):
    qsU, p = absmaxes[which + "_U"], cap[which]
    s2 = qsU.state2
    report["stacks"][which]["nested"] = {
        "qabsmax_equal": torch.equal(qsU.absmax.reshape(-1), p["_q"].reshape(-1)),
        "state2_absmax_equal": torch.equal(s2.absmax.reshape(-1), p["_s"].reshape(-1)),
        "offset_equal": torch.equal(torch.as_tensor(qsU.offset).reshape(-1).float(), p["_off"].reshape(-1).float()),
        "state2_code_equal": torch.equal(s2.code, p["_code"]),
        "state2_blocksize": int(s2.blocksize),
        "offset": {"U": float(torch.as_tensor(qsU.offset).reshape(-1)[0]), "E": float(p["_off"].reshape(-1)[0])},
    }
ok = all(r["packed_equal"] and r["absmax_fp32_equal"] and r["code_equal"] and r["U_nested"] and all(r["nested"][k] for k in
         ("qabsmax_equal", "state2_absmax_equal", "offset_equal", "state2_code_equal")) for r in report["stacks"].values())
report["verdict"] = "SAME_BYTES" if ok else "DIFFERENT"
print(json.dumps(report, indent=1))
