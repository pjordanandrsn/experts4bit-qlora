"""DQ11 MLP-adapter BUILD Phase 2 (``DQ11-MLP-BUILD-AMENDMENT-5.md``): one process per part, then a CPU reduce.

Parts:
- ``stage-check``: the target's lock, the model snapshot hashed, the sealed inputs.
- ``b-read``: (b) at the target's lock on the stratified draws (``dq11_mlp_phase2.read_layers``).
- ``F0a F1a F1b F0b F0c F1c F1d F0d``: the timed arms, ABBA twice, each a fresh process of ``updates`` updates of the
  shipped loop (warmup 5, so T_arm = the median of updates 6..N). F1 is the engine (``E4B_DENSE_MLP_LORA_FUSE=1``); F0
  never imports it. F0a and F1a are also scored at the start and the end (QUALITY). After the timed loop, each arm runs
  one untimed forward and backward under the LoRA forward counter (the engagement witness).
- ``D0 D1``: the drift arms, untimed, under configuration C (deterministic settings, not the timed configuration):
  layer 0 ``up_proj``'s A and B after updates 1, 2, 4, 8, 16, 32 and 40, REPORTED only.
- ``reduce``: section 7's rungs over the stage check, the (b) part and the eight timed arms. The drift records are
  read only for the report, and their absence or failure never changes the outcome.

Targets:
- ``a2000``: the $0 shakedown (the rehearsal lock, 8 updates). It is never Phase 2 data.
- ``5090``: the read (the science lock, 40 updates).

The static fit, (boot + the parts' seconds) x 1.25 <= the guard, is :func:`static_fit`."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dq11_mlp_phase1_run as r1  # noqa: E402

SCHEMA = "dq11-mlp-phase2/1"
TIMED = ("F0a", "F1a", "F1b", "F0b", "F0c", "F1c", "F1d", "F0d")       # ABBA twice (the PREREG's order)
DRIFT = ("D0", "D1")
PARTS = ("stage-check", "b-read") + TIMED + DRIFT
QUALITY_ARMS = ("F0a", "F1a")
PAIRS = (("F0a", "F1a"), ("F0b", "F1b"), ("F0c", "F1c"), ("F0d", "F1d"))   # D4: each block pair's F0 and F1
SELF_PAIRS = (("F0a", "F0b"), ("F1a", "F1b"), ("F0c", "F0d"), ("F1c", "F1d"))
NOISY_BAND = (0.985, 1.015)
H_BAR = 0.05
QUALITY_BAND = 0.05
WARMUP = 5
DRIFT_AT = (1, 2, 4, 8, 16, 32, 40)
DRIFT_SLOT = "layers.0.mlp.up_proj"
GUARD_S = 2 * 3600
BOOT_S = 5 * 60
FIT_FACTOR = 1.25
TARGETS = {
    "a2000": {"lock": "rehearsal/wheels.json", "gpu": "NVIDIA RTX A2000 12GB", "capability": (8, 6), "cuda": "12.8",
              "updates": 8, "free_min_gib": 9, "science_eligible": False},
    "5090": {"lock": "wheels.json", "gpu": "NVIDIA GeForce RTX 5090", "capability": (12, 0), "cuda": "13.0",
             "updates": 40, "free_min_gib": 20, "science_eligible": True},
}


def static_fit(part_seconds, *, boot_s=BOOT_S, guard_s=GUARD_S, factor=FIT_FACTOR) -> dict:
    """Amendment 5 section 8: (boot + the parts' seconds) x 1.25 <= the guard."""
    need = (boot_s + sum(part_seconds)) * factor
    return {"boot_s": boot_s, "parts_s": sum(part_seconds), "factor": factor, "need_s": need, "guard_s": guard_s,
            "fits": need <= guard_s}


def require_runtime(target, directory, cache) -> dict:
    """Refuse anything but the target's card under the target's lock, with the bus claim confirmed."""
    import torch

    t = TARGETS[target]
    if os.environ.get("DQ11_GPU_CLAIM_CONFIRMED") != "1":
        raise RuntimeError("PHASE2_REFUSAL: the GPU claim is not confirmed")
    lock_path = HERE / t["lock"]
    if r1._sha(Path(directory) / "wheels.json") != r1._sha(lock_path):
        raise RuntimeError(f"PHASE2_REFUSAL: the workspace's wheels.json is not {t['lock']}")
    rows = {r["name"].lower(): r for r in json.loads(lock_path.read_text())["packages"]}
    wheels = {name: r1.wheel_identity(name, rows[name], cache) for name in r1.LOCKED}
    bad = [n for n, w in wheels.items() if w["installed"] != w["locked"] or w["wheel_sha256"] != w["lock_sha256"]
           or not w["files_match_wheel"]]
    if bad:
        raise RuntimeError(f"PHASE2_REFUSAL: not the lock's install: {bad}")
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != t["gpu"]
            or torch.cuda.get_device_capability() != t["capability"] or torch.version.cuda != t["cuda"]):
        raise RuntimeError(f"PHASE2_REFUSAL: target {target} runs on one {t['gpu']} (cuda {t['cuda']}) only")
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    if free < t["free_min_gib"] * (1 << 30):
        raise RuntimeError(f"PHASE2_REFUSAL: {free / (1 << 30):.1f} GiB free, under {t['free_min_gib']}")
    if target == "a2000":
        cap = min(9 * (1 << 30), int(total * 0.75))
        torch.cuda.set_per_process_memory_fraction(cap / total)
    return {"target": target, "wheels": wheels, "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
            "free_before_bytes": free, "e4b_sha": os.environ.get("E4B_SHA")}


def _train(model, trainable, directory, workload, tokens, locked, updates, *, warmup) -> dict:
    """The shipped loop (``loggetta``'s ``train_loop``) for ``updates`` updates; its own per-update wall times (CUDA
    synchronized at each end) and losses."""
    from loggetta.backends import dense_train
    from loggetta.backends.experts4bit_train import train_loop
    from loggetta.measure import DriverMemorySampler

    data = SimpleNamespace(blocks=tokens["train"], info={"source": "DQ11 sealed tokens", "tokens": locked["tokens"]["train"]})
    meas = {}
    with DriverMemorySampler() as sampler:
        out = train_loop(model, trainable, str(Path(directory) / "hf-cache/model"), replace(workload, steps=updates),
                         sampler, meas, seed=3407, warmup=warmup, prepared_data=data,
                         frozen_digest=dense_train.frozen_digest, frozen_kind="dense")
    if out.get("status") != "OK":
        raise RuntimeError("PHASE2_REFUSAL: the shipped loop failed its integrity check")
    return {"losses": out["correctness"]["losses"], "step_seconds": meas["step_seconds"],
            "timed_steps": meas["timed_steps"], "device_peak_bytes": meas.get("device_peak_bytes")}


def _witness(model, ids) -> dict:
    """One untimed forward and backward under the LoRA forward counter: which PEFT LoRA modules ran."""
    import dq11_mlp_phase1 as p1

    with p1.lora_forward_counter(model) as counts:
        model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    model.zero_grad(set_to_none=True)
    counts = dict(counts)
    mlp = sum(v for k, v in counts.items() if k.rsplit(".", 1)[-1] in p1.PROJECTIONS)
    return {"mlp_lora_forwards": mlp, "attention_lora_forwards": sum(counts.values()) - mlp}


class _Snapshots:
    """Layer 0 ``up_proj``'s A and B after the updates in :data:`DRIFT_AT`, through torch's global optimizer step
    post-hook (the shipped loop is not edited)."""

    def __init__(self, slots, updates):
        self.slots, self.want, self.step, self.saved = slots, {t for t in DRIFT_AT if t <= updates}, 0, {}

    def __call__(self, _opt, _args, _kwargs):
        self.step += 1
        if self.step in self.want:
            for s in ("A", "B"):
                self.saved[f"step{self.step}.{s}"] = self.slots[f"{DRIFT_SLOT}.{s}"].detach().float().cpu().clone()


def run_part(target, part, directory, cache, out) -> dict:
    from dq11_arm import read_inputs
    from dq11_proof_policy import process_environment

    t = TARGETS[target]
    record = {"schema": SCHEMA, "part": part, "target": target, "science_eligible": t["science_eligible"],
              "environment": "C" if part in DRIFT else "read"}
    with process_environment("proof" if part in DRIFT else "read"):
        record["runtime"] = require_runtime(target, directory, cache)
        record["model_files"] = r1.verify_model(directory, hashes=part == "stage-check")
        if part == "stage-check":
            locked, _tokens = read_inputs(Path(directory))
            if r1._sha(Path(directory) / "locked_inputs.json") != r1._sha(HERE / "locked_inputs.json"):
                raise RuntimeError("PHASE2_REFUSAL: the workspace's locked_inputs.json is not the repository's")
            record["inputs"] = {"locked_inputs_sha256": r1._sha(Path(directory) / "locked_inputs.json"),
                                "assets": locked["assets"], "status": "PASS"}
            return record
        import torch

        fused = part in ("b-read", "D1") or part.startswith("F1")
        if fused and part != "b-read":
            os.environ["E4B_DENSE_MLP_LORA_FUSE"] = "1"
        else:
            os.environ.pop("E4B_DENSE_MLP_LORA_FUSE", None)
        locked, tokens = read_inputs(Path(directory))
        model, trainable, workload, built = r1.build_l(directory)
        record["built"] = {"initial_sha256": built["initial_sha256"]}
        layers = r1.decoder_mlps(model)
        ids = torch.tensor([tokens["train"][0]], device="cuda", dtype=torch.long)
        if part == "b-read":
            import dq11_mlp_phase1 as p1
            import dq11_mlp_phase2 as p2
            from experts4bit_qlora.engines import dense_lora_fuse as dlf

            checked = {i: layers[i] for i in r1.LAYERS}
            xs, gys, record["capture_loss"] = r1.capture(model, ids, checked)
            record["read"] = p2.read_layers(
                {i: checked[i].mlp for i in r1.LAYERS}, xs, gys,
                enable=lambda: dlf.enable_dense_mlp_lora_fuse(model), disable=lambda: dlf.disable_dense_mlp_lora_fuse(model),
                stats=dlf.DENSE_MLP_LORA_STATS, counter=lambda: p1.lora_forward_counter(model),
                stage=lambda i: checked[i]._dense_offload.stage(), evict=lambda i: checked[i]._dense_offload.evict(),
                adapters=r1.ADAPTERS)
            return record
        stats = None
        if fused:
            from experts4bit_qlora.engines import dense_lora_fuse as dlf

            record["knob_requested"] = dlf.dense_mlp_lora_fuse_requested()
            record["enabled_adapters"] = dlf.enable_dense_mlp_lora_fuse(model)
            record["refused"] = dict(dlf.DENSE_MLP_LORA_STATS["refused"])
            stats = dlf.DENSE_MLP_LORA_STATS
        if part in DRIFT:
            from dq11_common import adapter_slots
            from dq11_proof_policy import observer_policy
            from safetensors.torch import save_file

            snaps = _Snapshots(adapter_slots(model)[0], t["updates"])
            handle = torch.optim.optimizer.register_optimizer_step_post_hook(snaps)
            try:
                with observer_policy() as policy:
                    record["training"] = _train(model, trainable, directory, workload, tokens, locked, t["updates"],
                                                warmup=0)
            finally:
                handle.remove()
            record["policy"] = policy
            path = Path(out) / f"drift-{part}.safetensors"
            save_file(snaps.saved, str(path))
            record["snapshots"] = {"file": path.name, "sha256": r1._sha(path), "steps": sorted(snaps.want),
                                   "slot": DRIFT_SLOT}
        else:
            if part in QUALITY_ARMS:
                from dq11_common import score

                record["quality_initial"] = score(model, tokens)
            record["training"] = _train(model, trainable, directory, workload, tokens, locked, t["updates"], warmup=WARMUP)
            if part in QUALITY_ARMS:
                record["quality_final"] = score(model, tokens)
            record["witness"] = _witness(model, ids)
        if stats is not None:
            record["engine_stats"] = {k: stats.get(k) for k in ("fused", "adapters", "calls", "fallback")}
        record["engine_imported"] = "experts4bit_qlora.engines.dense_lora_fuse" in sys.modules
    return record


# --- the reduce (CPU) -------------------------------------------------------------------------------------------------

def t_arm(rec) -> float:
    """The median of updates 6..N (unrounded)."""
    s = rec["training"]["step_seconds"]
    return statistics.median(s[WARMUP:] or s)


def engagement(records) -> list:
    """Rung 1's failures: the (b) part's engagement, and every F1 arm's (96 engaged, nothing refused, fallback 0, no PEFT
    MLP LoRA forward in the witness); an F0 arm that imported the engine."""
    why = []
    e = records["b-read"]["read"].get("engagement", {})
    if not e.get("ok"):
        why.append(f"b-read: {e}")
    for a in TIMED:
        r = records[a]
        if a.startswith("F1"):
            st = r.get("engine_stats") or {}
            if (r.get("enabled_adapters") != r1.ADAPTERS or r.get("refused") or st.get("fallback")
                    or r["witness"]["mlp_lora_forwards"]):
                why.append(f"{a}: enabled {r.get('enabled_adapters')}, refused {r.get('refused')}, "
                           f"fallback {st.get('fallback')}, PEFT MLP LoRA forwards {r['witness']['mlp_lora_forwards']}")
        elif r.get("engine_imported") or r["witness"]["mlp_lora_forwards"] == 0:
            why.append(f"{a}: the unfused arm imported the engine or ran no PEFT MLP LoRA forward")
    return why


def quality(records) -> dict:
    out, ok = {}, True
    f0, f1 = records["F0a"], records["F1a"]
    for when in ("quality_initial", "quality_final"):
        for text in ("alpaca-heldout", "wikitext-test"):
            d = f1[when][text]["ppl"] - f0[when][text]["ppl"]
            out[f"{when}:{text}"] = d
            ok = ok and abs(d) <= QUALITY_BAND
    return {"deltas": out, "ok": ok}


def drift_report(directory) -> dict:
    """REPORTED only. Reads the drift records if both are there; never raises."""
    try:
        recs = {a: json.loads((Path(directory) / f"{a}.json").read_text()) for a in DRIFT}
        from safetensors.torch import load_file

        snap = {a: load_file(str(Path(directory) / recs[a]["snapshots"]["file"])) for a in DRIFT}
        rows = {}
        for key in sorted(snap["D0"], key=lambda k: (int(k.split(".")[0][4:]), k)):
            a, b = snap["D0"][key].double(), snap["D1"][key].double()
            n = float(a.norm())
            rows[key] = float((b - a).norm()) / n if n > 0 else (0.0 if float(b.norm()) == 0 else float("inf"))
        l0, l1 = recs["D0"]["training"]["losses"], recs["D1"]["training"]["losses"]
        return {"label": "under C (deterministic settings), not the timed configuration",
                "relative_distance": rows, "max_abs_dloss": max(abs(x - y) for x, y in zip(l0, l1))}
    except Exception as e:                                              # noqa: BLE001 - REPORTED only, never a ruling
        return {"label": "drift: not read", "reason": f"{type(e).__name__}: {e}"}


def reduce(directory) -> dict:
    """Amendment 5 section 7. The first rung that applies decides."""
    d = Path(directory)
    records = {p: json.loads((d / f"{p}.json").read_text()) for p in ("stage-check", "b-read") + TIMED}
    target = records["stage-check"]["target"]
    read = records["b-read"]["read"]
    b = read.get("outcome") or {}
    t = {a: t_arm(records[a]) for a in TIMED}
    selfs = {f"{y}/{x}": t[y] / t[x] for x, y in SELF_PAIRS}
    noisy = [k for k, v in selfs.items() if not NOISY_BAND[0] <= v <= NOISY_BAND[1]]
    h = {f"{f1}/{f0}": (t[f0] - t[f1]) / t[f0] for f0, f1 in PAIRS}
    q = quality(records)
    eng = engagement(records)
    out = {"schema": SCHEMA, "target": target, "shakedown": target == "a2000", "t_arm": t, "self_pairs": selfs,
           "h": h, "h_min": min(h.values()), "quality": q, "b": b, "b_void": read.get("void"),
           "drift": drift_report(d), "science_eligible": TARGETS[target]["science_eligible"]}
    if eng or read.get("void"):
        out["outcome"] = "VOID (harness): " + "; ".join(([read["void"]] if read.get("void") else []) + eng)
    elif noisy:
        out["outcome"] = "VOID (NOISY, a strike): " + " ".join(noisy)
        if b.get("fail"):
            out["outcome"] += "; ACCURACY_FAIL: " + " ".join(b["fail"])
    elif b.get("fail"):
        out["outcome"] = "ACCURACY_FAIL: " + " ".join(b["fail"])
    elif b.get("calibration_indeterminate"):
        out["outcome"] = "NO_GAIN (accuracy indeterminate: " + " ".join(b["calibration_indeterminate"]) + ")"
    elif b.get("unjudged_at_k"):
        out["outcome"] = "NO_GAIN (accuracy unjudged: " + " ".join(b["unjudged_at_k"]) + ")"
    else:
        failed = ([f"H {out['h_min']:.6f} < {H_BAR}"] if out["h_min"] < H_BAR else []) + ([] if q["ok"] else ["QUALITY_FAIL"])
        out["outcome"] = "GAIN" if not failed else "NO_GAIN: " + "; ".join(failed)
    out["named"] = {key: b[key] for key in ("fail", "calibration_indeterminate", "unjudged_at_k", "rank1_unresolved",
                                            "accuracy_over_reorder") if b.get(key)}
    if out["shakedown"]:
        out["outcome"] = "SHAKEDOWN (never Phase 2 data): " + out["outcome"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--part", required=True, choices=PARTS + ("reduce",))
    ap.add_argument("--directory", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.part == "reduce":
        rec = reduce(out)
        if rec["target"] != a.target:
            raise SystemExit(f"PHASE2_REFUSAL: the records are target {rec['target']}, not {a.target}")
        (out / "phase2.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        print(rec["outcome"])
        return 0
    rec = run_part(a.target, a.part, a.directory, a.cache, out)
    (out / f"{a.part}.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
