"""The PREREG's two arms, on the pod. Writes receipts incrementally.

Everything here is graded against `bench/mxfp4-arena-train/PREREG.md` as amended
twice. Predictions P1-P4, gates G1-G4. Nothing retries; a failure is the result.

Run as:  python v4_run.py <stage>   where stage in {gates, stock, arena}
Split into stages so a failure in one is a receipt rather than a lost run, and so
the expensive STOCK load cannot take the ARENA arm down with it.
"""
import json
import os
import sys
import time
import traceback

import torch

CKPT = os.environ.get("V4_CKPT", "/root/ckpt")
ARENA = os.environ.get("V4_ARENA", "/root/v4.arena")
OUT = os.environ.get("V4_OUT", "/root/v4out")
MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
VRAM_CAP_GIB = 24.0          # the PREREG's claim is "one 24 GB card"

os.makedirs(OUT, exist_ok=True)


def emit(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    print(f"[receipt] {name}: {json.dumps(obj, default=str)[:600]}")


def cap_vram():
    """Make a bigger card faithfully test the 24 GB claim.

    The claim under test is "runs on a single 24 GB card". The ladder may land on
    a 48 GB shape when both 24 GB shapes are out of stock or wedged, and running
    there unconstrained would answer a question nobody asked. Capping the process
    to 24 GiB makes the larger card a valid instrument instead of a weaker one —
    an allocation past the cap raises, exactly as it would on the real thing.
    """
    free, total = torch.cuda.mem_get_info()
    total_gib = total / 2**30
    if total_gib > VRAM_CAP_GIB + 1:
        torch.cuda.set_per_process_memory_fraction(VRAM_CAP_GIB / total_gib)
        return {"card_gib": round(total_gib, 1), "capped_to_gib": VRAM_CAP_GIB,
                "note": "card is larger than the claim; process capped so the test is honest"}
    return {"card_gib": round(total_gib, 1), "capped_to_gib": None,
            "note": "card is at or under the claim; no cap needed"}


# --------------------------------------------------------------------- gates
def stage_gates():
    """G2/G4 in the IMPORTED module, before anything expensive is graded."""
    import experts4bit_qlora as e4b
    from experts4bit_qlora.engines import fast, nvme_experts, nvme_train
    from experts4bit_qlora import lora as loramod
    import mxfp4_grouped
    import mxfp4_residency

    # Test BEHAVIOUR where it is cheap, and the compiled code otherwise. The
    # first version of this gate grepped `_eligible.__doc__` for the marker; the
    # marker lives in the function's CODE, so the gate reported "overlay not in
    # the imported module" against a correctly overlaid pod and killed the run
    # before the download. A gate that greps prose fails whenever someone rewords
    # a comment, which is the opposite of what a gate is for.
    class _NotCuda:
        is_cuda = False

    class _StubMxfp4Base:
        # Complete enough that the STOCK `_eligible` also returns cleanly. Without
        # `gate_up_proj` the un-overlaid version fell through to
        # `mod.gate_up_proj.is_cuda` and raised AttributeError -- still a failed
        # gate, but reported as a crash instead of "the refusal is absent".
        _e4b_mxfp4_arena = True
        bits, quant_type, blocksize = 4, "nf4", 64
        _gate_up_shape, _down_shape = (4096, 4096), (4096, 2048)
        gate_up_proj = _NotCuda()

    try:
        fast_reason = fast._eligible(_StubMxfp4Base())
    except Exception as exc:                     # a stub gap must not read as a verdict
        fast_reason = f"STUB-INCOMPATIBLE: {type(exc).__name__}: {exc}"

    g = {
        "e4b_version": e4b.__version__,
        "gnf4_mxfp4_grouped": hasattr(mxfp4_grouped, "gemm_mxfp4_grouped"),
        "V4_RESIDENCY_KINDS": list(mxfp4_residency.V4_RESIDENCY_KINDS),
        # G2 — the overlay is present in the imported module, per file
        "overlay_mxfp4_forward": hasattr(nvme_experts, "mxfp4_experts_forward"),
        "overlay_mxfp4_dequant": hasattr(nvme_experts, "_mxfp4_dequantize_expert"),
        "overlay_fused_lane": hasattr(nvme_experts, "_mxfp4_fused_forward"),
        "overlay_arena_view": hasattr(nvme_train, "arena_offload_view"),
        # functional: an MXFP4-flagged module must be REFUSED, and for that reason
        "overlay_fast_refusal": bool(fast_reason) and "mxfp4" in fast_reason.lower(),
        "overlay_fast_refusal_reason": fast_reason,
        # compiled-code check: instantiating an ExpertsLoRA here would need real
        # tensors, so read the constant the branch tests rather than the prose
        "overlay_gemv_bail": "_e4b_mxfp4_arena" in
            loramod.ExpertsLoRA._use_infer_gemv.__code__.co_consts,
        # The gnf4 half of the overlay (PR #75, merged 0f68952). Without it a real
        # V4 arena cannot be staged at all -- this is precisely what stopped the
        # previous attempt, one call after the model had loaded.
        "overlay_gnf4_reads_e8m0":
            __import__("nvme_residency")._ST_TO_TORCH.get("F8_E8M0") == "uint8",
    }
    # `_reason` is diagnostic text, not a check — keep it out of the verdict
    g["G2"] = all(g[k] for k in list(g)
                  if k.startswith("overlay_") and not k.endswith("_reason"))
    emit("gates_pre.json", g)
    if not g["G2"]:
        raise SystemExit("G2 FAILED: the overlay is not in the imported module")
    return g


# --------------------------------------------------------------------- STOCK
def stage_stock():
    """P1's control: RUN, not asserted. Its traceback is the receipt.

    Amendment 1 respecified this arm: NO quantization config, let transformers
    attempt to materialize the model. A config-class rejection is not evidence of
    a memory ceiling, which is how attempt 1's control failed for the wrong reason.
    """
    rec = {"arm": "STOCK", "started": time.time()}
    try:
        from transformers import AutoModelForCausalLM
        t0 = time.time()
        AutoModelForCausalLM.from_pretrained(
            CKPT, dtype=torch.bfloat16, trust_remote_code=True, device_map={"": 0})
        rec.update(outcome="LOADED", seconds=round(time.time() - t0, 1),
                   note="STOCK SUCCEEDED — P1 is REFUTED and that is the headline")
    except BaseException as exc:                    # noqa: BLE001 - the traceback IS the result
        rec.update(outcome="FAILED", error_type=type(exc).__name__,
                   error=str(exc)[:2000], traceback=traceback.format_exc()[-6000:])
        low = (type(exc).__name__ + " " + str(exc)).lower()
        rec["failed_on_memory"] = any(
            s in low for s in ("out of memory", "outofmemory", "cannot allocate",
                               "not enough memory", "killed", "oom"))
    emit("arm_stock.json", rec)
    return rec


# --------------------------------------------------------------------- ARENA
def stage_arena(steps=3, tokens=1, seqlen=256):
    tag = f"t{tokens}_s{seqlen}"
    from experts4bit_qlora import load_moe_4bit_streaming
    from experts4bit_qlora.engines.fast import enable_fast_train
    from experts4bit_qlora.engines.nvme_train import enable_nvme_train_residency
    from experts4bit_qlora.lora import ExpertsLoRA
    import nvme_residency

    rec = {"arm": "ARENA", "tokens": tokens, "seqlen": seqlen,
           "total_tokens": tokens * seqlen, "steps": steps,
           "vram": cap_vram()}
    torch.cuda.reset_peak_memory_stats()

    # hot_rows from MEASURED capacity, never a declared figure.
    from nvme_arena import load_index
    index = load_index(ARENA)
    stride = int(index["row_stride"])
    # MEASURED free RAM, from the CGROUP -- `free`/psutil report the HOST's
    # memory on a pod (256 cores and 1 TB were visible where the real limits were
    # 27.2 CPUs and 125 GB). A hot_rows sized off the host figure would OOM
    # partway through the first step.
    #
    # PAGE CACHE IS NOT IN USE. cgroup v2 counts it in `memory.current`, so the
    # naive `memory.max - memory.current` read **18.3 MB** right after the 138 GiB
    # arena bake had filled the cache -- capacity_for_bytes then returned ~nothing
    # and hot_rows fell through to an unvalidated floor. `memory.stat`'s `file`
    # field is that reclaimable cache; subtract it before calling the remainder
    # free.
    mem = {}
    usable = None
    try:
        lim = open("/sys/fs/cgroup/memory.max").read().strip()
        cur = int(open("/sys/fs/cgroup/memory.current").read().strip())
        stat = {}
        for line in open("/sys/fs/cgroup/memory.stat"):
            k, _, v = line.partition(" ")
            stat[k] = int(v)
        cache = stat.get("file", 0)                   # reclaimable page cache
        mem = {"limit": None if lim == "max" else int(lim), "current": cur,
               "file_cache": cache, "in_use": cur - cache}
        if lim != "max":
            usable = int((int(lim) - (cur - cache)) * 0.6)   # headroom for the run
    except Exception as exc:
        mem = {"error": str(exc)}
    # A sane floor: if the cgroup read produced something implausible, say so in
    # the receipt rather than sizing a 38 GiB pinned arena off it silently.
    if usable is None or usable < int(2e9):
        mem["rejected"] = f"cgroup usable={usable}; falling back to 24 GB"
        usable = int(24e9)
    rec["host_mem"] = mem
    try:
        # capacity_for_bytes(usable_bytes, row_stride, *, pinned=True) -- TWO
        # positionals. Calling it with one raised, and the fallback silently
        # handed back 4000 rows = 53 GB of pinned DRAM off a 12.75 MiB stride.
        rows = int(nvme_residency.capacity_for_bytes(usable, stride, pinned=True))
    except Exception as exc:                                  # older gnf4
        rows = max(1, int(usable // (stride * 2)))
        rec["capacity_for_bytes"] = f"unavailable ({exc}); computed {rows} directly"
    # Hard floor: at least the distinct cold experts one forward can want.
    # The FLOOR is a correctness requirement (every distinct cold expert one
    # forward routes must fit), the capacity is a resource limit. If the floor
    # exceeds capacity that is a refusal, not a max() -- silently taking the
    # floor is how 38.2 GiB of pinned DRAM got requested off an 18 MB reading.
    floor = min(int(tokens * seqlen * 6), int(index["n_experts_per_layer"]))
    cap = min(rows, 20000)
    if floor > cap:
        rec.update(hot_rows_floor=floor, hot_rows_capacity=cap,
                   outcome="HOT_ROWS_UNSATISFIABLE")
        emit(f"arm_arena_{tag}.json", rec)
        raise SystemExit(
            f"hot_rows floor {floor} exceeds measured capacity {cap}: this host "
            "cannot hold the rows one forward needs, and taking the floor anyway "
            "would pin more DRAM than exists")
    hot_rows = max(floor, cap)
    rec["usable_bytes_measured"] = usable
    rec["hot_rows_floor"] = floor
    rec["hot_rows_capacity"] = cap
    rec.update(row_stride=stride, hot_rows=hot_rows,
               arena_layers=index["n_layers"], arena_experts=index["n_experts_per_layer"])
    emit(f"progress_{tag}.json", rec)

    t0 = time.time()
    # CKPT, not MODEL_ID. The loader does
    # `snap = model_id if os.path.isdir(model_id) else snapshot_download(model_id)`,
    # so passing the hub id makes it download the 160 GB checkpoint a SECOND time
    # -- onto a disk already holding the first copy and the 138 GiB arena. That is
    # what killed attempt 2, as a xet writer error rather than an honest ENOSPC.
    model, cfg = load_moe_4bit_streaming(
        CKPT, "cuda", torch.bfloat16, r=8, alpha=16, quant_type="nf4",
        arena=ARENA, arena_train=True, trust_remote_code=True)
    rec["load_seconds"] = round(time.time() - t0, 1)
    rec["load_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    emit(f"progress_{tag}.json", rec)

    n = enable_nvme_train_residency(model, ARENA, hot_rows=hot_rows)
    rec["G1_modules_patched"] = n
    if n <= 0:
        emit("arm_arena.json", rec | {"outcome": "G1_FAILED"})
        raise SystemExit("G1 FAILED: 0 modules patched; every later number is meaningless")

    # G4 — the arm must be on the lane it claims. G1 counts patches, which no
    # longer implies which arithmetic runs.
    bases = [m.base for m in model.modules() if isinstance(m, ExpertsLoRA)]
    flagged = sum(1 for b in bases if getattr(b, "_e4b_mxfp4_arena", False))
    overridden = sum(1 for b in bases
                     if "_mxfp4" in getattr(getattr(b, "_dequantize_expert", None),
                                            "__name__", ""))
    fast_n = enable_fast_train(model)          # MUST be 0 (amendment 2)
    rec.update(G4_expertslora_modules=len(bases), G4_mxfp4_flagged=flagged,
               G4_mxfp4_dequant_override=overridden, G4_enable_fast_train_returned=fast_n)
    rec["G4"] = bool(bases) and flagged == len(bases) and overridden == len(bases) and fast_n == 0
    emit(f"progress_{tag}.json", rec)
    if not rec["G4"]:
        emit("arm_arena.json", rec | {"outcome": "G4_FAILED"})
        raise SystemExit("G4 FAILED: the arm is not on the lane it claims")

    # G3 — gradient checkpointing is REQUIRED by the tier, not optional.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    rec["G3_grad_checkpointing"] = True

    # P3, first half: the frozen storage must not move. Sample deterministically —
    # hashing all 147 GB twice is not free, and what is at risk is the STAGED
    # buffers, not the read-only file.
    import hashlib

    def frozen_digest():
        h = {}
        for i, b in enumerate(bases):
            if i % 11:                       # ~4 layers of 43
                continue
            m = hashlib.sha256()
            for nm in ("gate_up_proj", "gate_up_absmax", "down_proj", "down_absmax"):
                t = getattr(b, nm)
                if t.numel():
                    m.update(t.detach().cpu().numpy().tobytes())
            h[f"layer_{i}"] = m.hexdigest()
        return h

    before = frozen_digest()

    params = [p for p in model.parameters() if p.requires_grad]
    rec["trainable_tensors"] = len(params)
    rec["trainable_params"] = int(sum(p.numel() for p in params))
    if not params:
        emit("arm_arena.json", rec | {"outcome": "NO_TRAINABLE_PARAMS"})
        raise SystemExit("no trainable parameters — the adapter did not attach")
    opt = torch.optim.AdamW(params, lr=1e-4)
    vocab = int(getattr(cfg, "vocab_size", 32000))

    losses, step_s = [], []
    g = torch.Generator(device="cpu").manual_seed(0)
    for s in range(steps):
        ids = torch.randint(0, vocab, (tokens, seqlen), generator=g).to("cuda")
        t1 = time.time()
        out = model(input_ids=ids, labels=ids)
        loss = out.loss if hasattr(out, "loss") else out[0]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_s.append(round(time.time() - t1, 2))
        losses.append(float(loss.detach()))
        rec.update(losses=losses, step_seconds=step_s,
                   peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        emit(f"progress_{tag}.json", rec)

    after = frozen_digest()
    rec["P3_frozen_bytes_unchanged"] = before == after
    rec["P3_layers_hashed"] = len(before)
    rec["P3_note"] = ("sampled every 11th MoE layer, all four staged expert tensors; "
                      "the arena FILE is opened read-only, so what is at risk and "
                      "what is checked here is the staged device-side storage")
    rec["P2_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    rec["P2_in_band_4_to_8"] = 4.0 <= rec["P2_peak_gib"] <= 8.0
    rec["P2_under_24"] = rec["P2_peak_gib"] < 24.0
    rec["P4_loss_finite_and_moves"] = (
        all(map(lambda x: x == x and abs(x) != float("inf"), losses))
        and len(set(losses)) > 1)
    rec["outcome"] = "TRAINED"
    emit(f"arm_arena_{tag}.json", rec)
    return rec


if __name__ == "__main__":
    stage = sys.argv[1]
    # `arena` takes the REGISTERED shape on argv so each rung of the batch ladder
    # runs as its OWN process: an OOM at one rung then loses that rung's receipt
    # only, not every rung measured before it.
    kw = {}
    if stage == "arena" and len(sys.argv) >= 4:
        kw = {"tokens": int(sys.argv[2]), "seqlen": int(sys.argv[3])}
        if len(sys.argv) >= 5:
            kw["steps"] = int(sys.argv[4])
    try:
        {"gates": stage_gates, "stock": stage_stock,
         "arena": stage_arena}[stage](**kw)
    except SystemExit:
        raise
    except BaseException:
        emit(f"stage_{stage}_{kw.get('tokens', 0)}x{kw.get('seqlen', 0)}_crash.json",
             {"stage": stage, "traceback": traceback.format_exc()[-8000:]})
        raise
