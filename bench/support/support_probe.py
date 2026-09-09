#!/usr/bin/env python3
"""support_probe.py -- one repeatable row of architecture-support evidence.

``SUPPORTED_ARCHITECTURES`` in ``experts4bit_qlora/loader.py`` is a **loader**
claim: it decides whether e4b accepts a checkpoint at all. The right evidence
for that claim is load + verify + one forward on a real published checkpoint.
It is NOT a claim about CUDA-graph capture or throughput, and this probe does
not pretend to make one -- capture is recorded as ``not_tested`` with the
reason whenever the device is CPU.

That distinction is why this exists. ``docs/ARCHITECTURE_SUPPORT.md`` was
assembled by hand on 2026-08-12 against grouped-nf4-gemm 0.8.3, and its own
warning -- *"a fixture is not a checkpoint"* -- had already been earned once:
an earlier sweep called four families broken, and three of the four loaded fine
against real checkpoints. The failures were in generated configs. Nothing in
the tree re-runs that sweep, so every row has been ageing since, and a claimed
family with no row (``gpt_oss`` is claimed and marked ``not_tested``) looks
exactly like a validated one from outside.

Three things this deliberately does differently from the doc it feeds:

1. **The evidence tier is derived from measured facts, not asserted.** The doc
   marks rows "real weights", which is true of `tiny-random-DeepseekV2` and of
   `Qwen3-30B-A3B` alike -- one is a real *checkpoint*, the other is a real
   *model*. The row records ``total_params``, ``hidden_size``, ``num_experts``
   and a ``tier`` derived from them by the rule in ``_tier``, so a reader who
   disagrees with the rule can recompute it from the same row.
2. **Every stage is a field, so a partial pass is a row rather than a crash.**
   A family that loads and fails to forward is a different fact from one that
   will not load, and both are more useful than a traceback. Same discipline as
   ``bench/train-parity-20260905/tp1/logs/tp1_train_smoke.py`` (D7: a refusal is
   a receipt).
3. **Versions are recorded, not assumed.** Mixing rows across 22 minor versions
   of the kernel package under test is how the current matrix became unreadable.

Exit codes: 0 all attempted stages ok; 3 refused by the loader (a legitimate
answer, and a row); 4 the checkpoint cannot exercise the claim (its geometry is
below the quantizer's blocksize -- a fact about the fixture, not the family);
5 a shard is not the length its own header declares (a broken copy, not a
broken family); 6 load fault; 7 verify failed; 8 forward failed.
A JSON row is written in every one of those cases.

Keeping 3 and 4 apart is the whole point of the exit-code split. Collapsed, a
tiny-random that is too small to quantize is indistinguishable from a family the
loader genuinely will not accept, and the second reads as a support gap.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

# The probe is a bench driver, run from a checkout with PYTHONPATH=$PWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TOY_PARAM_CEILING = 1_000_000_000
"""Below this many parameters a published checkpoint is a `toy`.

Not a threshold with deep meaning -- it separates the tiny-random and
TinyStories checkpoints (10^6-10^8) from every real MoE release (>= 10^9) by a
wide margin, so no real family sits near the boundary. `total_params` is in the
row precisely so this call can be re-litigated without re-running anything.
"""


def _versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name, mod in (("torch", "torch"), ("transformers", "transformers"),
                      ("bitsandbytes", "bitsandbytes"), ("gnf4", "nf4_grouped"),
                      ("e4b", "experts4bit_qlora")):
        try:
            out[name] = __import__(mod).__version__
        except Exception as e:                      # noqa: BLE001 -- absence is data
            out[name] = f"absent ({type(e).__name__})"
    return out


def _cfg_int(cfg, *names):
    """First present attribute among ``names``, looking through a text tower.

    Key spellings differ per family (`num_experts` vs `num_local_experts` vs
    `n_routed_experts`), and multimodal configs nest the language model, so a
    single spelling would silently report None for half the matrix.
    """
    for holder in (cfg, getattr(cfg, "text_config", None)):
        if holder is None:
            continue
        for n in names:
            v = getattr(holder, n, None)
            if isinstance(v, int):
                return v
    return None


def _tier(total_params: int | None) -> str:
    """Evidence tier from the one fact that actually separates the cases: size.

    An earlier version of this returned ``synthetic`` for any local path, on the
    reasoning that a directory is our own fixture and an HF id is a release.
    That is wrong on this fleet: the released 1.4 TB Kimi-K3 snapshot and the
    real 13 GB OLMoE-1B-7B both sit on the NAS as plain local directories, and
    would have been mislabelled as fixtures -- understating the strongest
    evidence available. Whether a checkpoint is *generated* is not visible from
    its path, so it is declared (``--provenance``) rather than guessed.
    """
    if total_params is None:
        return "unknown"
    return "toy" if total_params < TOY_PARAM_CEILING else "reference"


def _safetensors_integrity(model_dir: Path) -> dict:
    """Verify each shard against its OWN declared length.

    A safetensors file is: 8 bytes of little-endian header length N, then N bytes
    of JSON, then the tensor data. So the complete size is
    ``8 + N + max(end of every data_offsets range)`` -- computable from the file
    itself, with no network and no reference copy.

    This exists because a size-stability heuristic is not a completeness check.
    A LAN copy of OLMoE stalled for over a minute mid-transfer, the watcher read
    the flat size as "settled", and a probe started against a checkpoint that was
    still being written. Nothing about that is visible in a directory listing,
    and the resulting load failure would have been recorded as a fact about
    `olmoe`. Asking the file how long it should be is the check that actually
    answers the question.
    """
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        return {"status": "not_tested", "reason": "no *.safetensors in the directory"}
    bad = []
    for s in shards:
        actual = s.stat().st_size
        try:
            with s.open("rb") as f:
                n = int.from_bytes(f.read(8), "little")
                if n <= 0 or n > 200_000_000:
                    bad.append({"file": s.name, "why": f"implausible header length {n}"})
                    continue
                head = json.loads(f.read(n))
        except Exception as e:                              # noqa: BLE001
            bad.append({"file": s.name, "why": f"header unreadable: {type(e).__name__}: {e}"})
            continue
        end = 0
        for name, meta in head.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            off = meta.get("data_offsets")
            if isinstance(off, (list, tuple)) and len(off) == 2:
                end = max(end, int(off[1]))
        expected = 8 + n + end
        if actual != expected:
            bad.append({"file": s.name, "why": f"size {actual} != declared {expected} "
                                               f"(short by {expected - actual})"})
    if bad:
        return {"status": "error", "shards": len(shards), "bad": bad,
                "reason": "at least one shard is not the length its own header declares -- "
                          "the copy is incomplete or corrupt, so anything below would be an "
                          "artefact of the transfer rather than a fact about the family"}
    return {"status": "ok", "shards": len(shards),
            "note": "every shard matches the length its own header declares"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, or a local snapshot path")
    ap.add_argument("--revision", default=None, help="pin the checkpoint by commit sha")
    ap.add_argument("--out", required=True, help="where to write the JSON row")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--provenance", default="published", choices=("published", "generated"),
                    help="'generated' for a fixture we built; declared, because a generated "
                         "config is indistinguishable from a released one by path or content")
    ap.add_argument("--no-forward", action="store_true")
    ap.add_argument("--seq", type=int, default=16, help="tokens in the single forward")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    row: dict = {
        "model": args.model,
        "revision": args.revision,
        "source": "local-path" if os.path.exists(args.model) else "hf-id",
        "provenance": args.provenance,
        "device": device,
        "dtype": args.dtype,
        "probe": "bench/support/support_probe.py",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": _versions(),
        "stages": {},
    }
    if device == "cuda":
        row["gpu"] = torch.cuda.get_device_name(0)
        row["capability"] = list(torch.cuda.get_device_capability(0))

    def write(code: int) -> int:
        row["exit_code"] = code
        row["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(row, indent=1, sort_keys=False) + "\n")
        print(f"wrote {p} (exit {code})")
        return code

    def fail(stage: str, exc: BaseException, code: int) -> int:
        row["stages"][stage] = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc().strip().splitlines()[-1],
        }
        return write(code)

    # ── config ────────────────────────────────────────────────────────────────
    try:
        cfg = AutoConfig.from_pretrained(
            args.model, revision=args.revision, trust_remote_code=args.trust_remote_code)
    except Exception as e:                                  # noqa: BLE001
        return fail("config", e, 6)

    model_type = getattr(cfg, "model_type", None)
    row["model_type"] = model_type
    row["config"] = {
        "hidden_size": _cfg_int(cfg, "hidden_size"),
        "num_hidden_layers": _cfg_int(cfg, "num_hidden_layers"),
        "num_experts": _cfg_int(cfg, "num_experts", "num_local_experts", "n_routed_experts"),
        "top_k": _cfg_int(cfg, "num_experts_per_tok", "top_k", "moe_topk"),
    }
    row["stages"]["config"] = {"status": "ok"}

    # ── is this family claimed, and by which route? ───────────────────────────
    # Two routes reach the loader: a SUPPORTED_ARCHITECTURES entry, or a
    # compatible convention record. They are different claims -- a family
    # supported only by convention is not in the shipped list -- so the row says
    # which, rather than collapsing both to "supported".
    from experts4bit_qlora import loader as _loader

    claimed = model_type in _loader.SUPPORTED_ARCHITECTURES
    by_convention = False
    if not claimed:
        try:
            by_convention = bool(_loader._read_compatible_convention(model_type))
        except Exception:                                   # noqa: BLE001
            by_convention = False
    row["claim"] = {
        "in_supported_architectures": claimed,
        "reachable_by_convention": by_convention,
        "expert_attr": _loader.SUPPORTED_ARCHITECTURES.get(model_type),
    }
    row["stages"]["claim"] = {"status": "ok"}

    # ── can this checkpoint exercise the claim at all? ────────────────────────
    # ExpertsNbit refuses any sub-16-bit stack whose hidden_dim or intermediate_dim
    # is not a multiple of the blocksize, because blocks must tile each expert
    # exactly (_vendor/experts.py:223). Every published qwen3_5_moe tiny-random is
    # hidden_size=8 / moe_intermediate=32, so it refuses before the family is
    # tested -- a fact about the fixture that reads exactly like a fact about the
    # loader. It is knowable from the config alone, so check it BEFORE pulling
    # gigabytes of weights, and give it its own exit code so a reducer never
    # scores it as an unsupported family.
    #
    # The blocksize is read from the constructor's own default rather than
    # hardcoded, so this cannot drift away from the check it is predicting.
    import inspect

    from experts4bit_qlora._vendor.experts import ExpertsNbit

    blocksize = inspect.signature(ExpertsNbit.__init__).parameters["blocksize"].default
    inter = _cfg_int(cfg, "moe_intermediate_size", "intermediate_size")
    offenders = {n: v for n, v in (("hidden_size", row["config"]["hidden_size"]),
                                   ("moe_intermediate_size", inter))
                 if v is None or v % blocksize != 0}
    row["config"]["moe_intermediate_size"] = inter
    if offenders:
        row["stages"]["precondition"] = {
            "status": "blocked",
            "blocksize": blocksize,
            "offending_dims": offenders,
            "reason": (f"{', '.join(f'{n}={v}' for n, v in offenders.items())} not divisible by "
                       f"blocksize {blocksize}; ExpertsNbit would refuse before this family is "
                       f"exercised. This is a property of the CHECKPOINT, not of {model_type!r} "
                       f"support -- a bigger checkpoint is needed to answer the question."),
        }
        return write(4)
    row["stages"]["precondition"] = {"status": "ok", "blocksize": blocksize}

    # For a local directory, verify the shards before loading them. An HF-cached
    # model is checksummed by huggingface_hub on download; a directory we copied
    # over the LAN is not checked by anything.
    if os.path.isdir(args.model):
        integ = _safetensors_integrity(Path(args.model))
        row["stages"]["integrity"] = integ
        if integ["status"] == "error":
            return write(5)
    else:
        row["stages"]["integrity"] = {
            "status": "not_tested",
            "reason": "hf-id: huggingface_hub verifies its own downloads"}

    # ── load ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        model, loaded_cfg = _loader.load_moe_4bit_streaming(
            args.model, device, getattr(torch, args.dtype), args.r, args.alpha,
            trust_remote_code=args.trust_remote_code or None, revision=args.revision)
    except Exception as e:                                  # noqa: BLE001
        # A refusal is an answer, not a fault: the loader declining a family it
        # cannot represent safely is the behaviour we want, and it gets its own
        # exit code so a reducer can tell it from a crash.
        #
        # The type alone cannot decide this. loader.py raises bare RuntimeError
        # for BOTH the zero-expert-stacks refusal (":1163 -- Refusing to return a
        # model with zero quantized expert layers") and a real fault
        # ("unmaterialized meta tensors remain"). Classifying by type would file
        # the loader's most deliberate refusal as a crash, which is precisely
        # backwards -- that guard firing on a dense checkpoint is the loader
        # working. So it is anchored on the loader's own refusal vocabulary as
        # well. Message matching is brittle and I would rather not rely on it:
        # the better fix is a dedicated refusal exception in e4b, at which point
        # this reduces to a type check.
        REFUSAL_TYPES = ("MoEConventionError", "NotImplementedError", "ValueError",
                         "EpilogueContractError")
        REFUSAL_MARKERS = ("Refusing", "no fused expert stacks found")
        refusal = (type(e).__name__ in REFUSAL_TYPES
                   or any(m in str(e) for m in REFUSAL_MARKERS))
        row["stages"]["load"] = {
            "status": "refused" if refusal else "error",
            "refusal_basis": ("type" if type(e).__name__ in REFUSAL_TYPES
                              else "message" if refusal else "n/a"),
            "error": f"{type(e).__name__}: {e}",
            "seconds": round(time.time() - t0, 1),
        }
        return write(3 if refusal else 6)
    row["stages"]["load"] = {"status": "ok", "seconds": round(time.time() - t0, 1)}

    total = sum(p.numel() for p in model.parameters())
    row["total_params"] = total
    row["tier"] = _tier(total)
    row["tier_rule"] = f"total_params < {TOY_PARAM_CEILING} -> toy; else reference"

    # ── verify the experts are actually low-bit ───────────────────────────────
    # Without this the probe would report "loaded" for a checkpoint whose expert
    # stacks stayed bf16, which is the exact failure load_in_4bit=True has and
    # the reason this loader exists.
    try:
        from experts4bit_qlora.verify import verify_moe_4bit
        rep = verify_moe_4bit(model, strict=True)
        row["stages"]["verify"] = {
            "status": "ok",
            "n_quantized": rep["n_quantized"],
            "n_unquantized": rep["n_unquantized"],
            "quant_types": sorted({q["quant_type"] for q in rep["quantized"]}),
        }
    except Exception as e:                                  # noqa: BLE001
        return fail("verify", e, 7)

    # ── one forward ───────────────────────────────────────────────────────────
    if args.no_forward:
        row["stages"]["forward"] = {"status": "not_tested", "reason": "--no-forward"}
    else:
        try:
            ids = torch.arange(1, args.seq + 1, device=device).unsqueeze(0)
            t1 = time.time()
            with torch.no_grad():
                out = model(input_ids=ids, labels=ids)
            loss = float(out.loss) if getattr(out, "loss", None) is not None else None
            row["stages"]["forward"] = {
                "status": "ok",
                # The input is arange token ids, so this loss is meaningless as a
                # quality figure -- it exists to prove the forward completes and
                # produces finite numbers. Recorded with the caveat attached so a
                # reader cannot mistake 14.06 for a perplexity claim.
                "note": "loss is on synthetic arange token ids: a finiteness smoke "
                        "test, NOT a quality measure",
                "seconds": round(time.time() - t1, 2),
                "loss": None if loss is None else round(loss, 5),
                "loss_finite": None if loss is None else bool(loss == loss and abs(loss) != float("inf")),
                "logits_finite": bool(torch.isfinite(out.logits).all()),
            }
        except Exception as e:                              # noqa: BLE001
            return fail("forward", e, 8)

    # ── capture is a different claim, on different hardware ───────────────────
    row["stages"]["capture"] = {
        "status": "not_tested",
        "reason": ("CPU probe: CUDA-graph capture cannot be established here"
                   if device == "cpu" else "not attempted by this probe"),
    }

    bad = [k for k, v in row["stages"].items() if v["status"] == "error"]
    return write(0 if not bad else 6)


if __name__ == "__main__":
    raise SystemExit(main())
