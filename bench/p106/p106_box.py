# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P106, the box (bench/p106/PREREG-p106.md; e4b#928 follow-up). ONE process, ONE engine (serve_paged.build_engine,
P98's configuration), Qwen3.6-35B-A3B. transformers' four Gated DeltaNet functions are switched IN PLACE between the
kernels they resolved (flash-linear-attention + causal-conv1d) and their torch path (``gdn_toggle.GdnToggle``), so the
two paths share the weights, the engine, the host and the moment:

Q  quality. Wikitext windows (P98's K8 windows) through ``PagedModelRunner`` teacher-forced: ``--prompt`` tokens of
   chunked prefill, then ``--cont`` single-row decode steps. The torch path and the kernels run IN LOCKSTEP on two KV
   slots (each prefill chunk and each decode step on one path, then the other), and every position's logits are
   compared on fp32 log-probs: KL(torch || kernels), the next-token NLL under each, argmax agreement, bit-identity.
   The first ``--null-windows`` windows also carry a THIRD slot on the torch path again: the instrument's own floor.
M  the mutant: window 0 again, the kernels called with ``use_qk_l2norm_in_kernel=False`` (a wrong flag a caller could
   pass), against the torch path. The bar must catch it, or the gate is inert.
T  TTFT. One request at a time through the scheduler (``max_new`` 1), prompt lengths ``--ttft-lengths``, the two
   paths alternated torch-kernels / kernels-torch per round after one warm-up each, CUDA-synchronised wall time.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_toggle import GdnToggle  # noqa: E402
from p98_box import loaded_commit, run_workload, wikitext_prompts  # noqa: E402

BLOCK = 128


class LogitTap:
    """Keeps the logits of the model's last forward (the runner reads only the argmax)."""

    def __init__(self, model):
        self.model, self.inner, self.logits = model, model.forward, None

        def keep(*a, **k):
            out = self.inner(*a, **k)
            self.logits = out.logits
            return out

        model.forward = keep

    def close(self):
        self.model.forward = self.inner


def compare(ref, alt, targets):
    """Per row of two [n, V] logit blocks: KL(ref || alt) on fp32 log-probs, the NLL of ``targets`` under each, argmax
    agreement, and whether the two rows are bit-identical."""
    kl, nll_r, nll_a, agree, same = [], [], [], [], []
    for i in range(0, ref.shape[0], BLOCK):
        r, a = ref[i:i + BLOCK], alt[i:i + BLOCK]
        lr, la = r.float().log_softmax(-1), a.float().log_softmax(-1)
        t = targets[i:i + BLOCK, None]
        kl += (lr.exp() * (lr - la)).sum(-1).tolist()
        nll_r += (-lr.gather(-1, t)[:, 0]).tolist()
        nll_a += (-la.gather(-1, t)[:, 0]).tolist()
        agree += (lr.argmax(-1) == la.argmax(-1)).tolist()
        same += (r == a).all(-1).tolist()
    return {"kl": kl, "nll_ref": nll_r, "nll_alt": nll_a, "agree": agree, "identical": same}


def merge(parts):
    out = {k: [] for k in ("kl", "nll_ref", "nll_alt", "agree", "identical")}
    for p in parts:
        for k in out:
            out[k] += p[k]
    return out


def summarize(rows):
    n = len(rows["kl"])
    if not n:
        return {"n": 0}
    d = [a - r for a, r in zip(rows["nll_alt"], rows["nll_ref"])]
    return {"n": n, "mean_kl": sum(rows["kl"]) / n, "max_kl": max(rows["kl"]),
            "median_kl": statistics.median(rows["kl"]), "argmax_agree": sum(rows["agree"]) / n,
            "identical_rows": sum(rows["identical"]), "mean_nll_ref": sum(rows["nll_ref"]) / n,
            "mean_nll_alt": sum(rows["nll_alt"]) / n, "mean_d_nll": sum(d) / n}


def lockstep(runner, tap, toggle, window, *, prompt, cont, chunk, paths, rid0):
    """``window`` (prompt + cont + 1 tokens) through every path in ``paths`` (name -> toggle setting, or a callable that
    sets it), each on its own KV slot, chunk by chunk and step by step. Returns {name: {"prefill": logits rows ...}}.
    Only the comparison rows against the first path are kept (``compare`` blocks), never the full distributions."""
    names = list(paths)
    rid = {nm: rid0 + i for i, nm in enumerate(names)}
    for i, nm in enumerate(names):
        runner.bind(rid[nm], i, window[:prompt])
    tgt = torch.tensor(window, device=runner.device)
    pre = {nm: [] for nm in names[1:]}
    dec = {nm: [] for nm in names[1:]}
    try:
        for start in range(0, prompt, chunk):
            take = min(chunk, prompt - start)
            got = {}
            for nm in names:
                set_path(toggle, paths[nm])
                runner.run_prefill([(rid[nm], start, take)])
                got[nm] = tap.logits[0]
            for nm in names[1:]:
                pre[nm].append(compare(got[names[0]], got[nm], tgt[start + 1:start + take + 1]))
            del got
        for nm in names:
            runner.tokens[rid[nm]][-1] = window[prompt]
        for t in range(cont):
            got = {}
            for nm in names:
                set_path(toggle, paths[nm])
                runner.run_decode([rid[nm]])
                got[nm] = tap.logits[:, -1]
                runner.tokens[rid[nm]][-1] = window[prompt + t + 1]
            for nm in names[1:]:
                dec[nm].append(compare(got[names[0]], got[nm], tgt[prompt + t + 1:prompt + t + 2]))
            del got
    finally:
        for nm in names:
            runner.free_slot(rid[nm])
        set_path(toggle, "resolved")
    return {nm: {"prefill": merge(pre[nm]), "decode": merge(dec[nm])} for nm in names[1:]}


def set_path(toggle, how):
    if callable(how):
        how(toggle)
    else:
        toggle.use(how)


def mutant(toggle):
    """The kernels with ``use_qk_l2norm_in_kernel=False`` forced on both delta rules (the conv functions untouched)."""
    toggle.use("resolved")
    for key, cells, resolved, _ in toggle.entries:
        if key.endswith("gated_delta_rule"):
            impl = resolved[0]

            def forced(*a, _impl=impl, **k):
                k["use_qk_l2norm_in_kernel"] = False
                return _impl(*a, **k)

            cells["implementation"].cell_contents = forced


def ttft(sched, toggle, prompts_by_len, rounds):
    """Median wall time from request to its one token, per prompt length and path; paths alternate per round."""
    out = {}
    for length, prompts in prompts_by_len.items():
        times = {"torch": [], "resolved": []}
        for path in ("torch", "resolved"):            # warm-up, untimed (fla's Triton autotune per shape)
            toggle.use(path)
            run_workload(sched, [prompts[0]], [1])
        for r in range(rounds):
            order = ("torch", "resolved") if r % 2 == 0 else ("resolved", "torch")
            for path in order:
                toggle.use(path)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                run_workload(sched, [prompts[1 + r]], [1])
                torch.cuda.synchronize()
                times[path].append((time.perf_counter() - t0) * 1e3)
        toggle.use("resolved")
        med = {p: statistics.median(v) for p, v in times.items()}
        out[str(length)] = {"ms": {p: [round(x, 2) for x in v] for p, v in times.items()},
                            "median_ms": {p: round(m, 2) for p, m in med.items()},
                            "speedup": round(med["torch"] / med["resolved"], 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--null-windows", type=int, default=2)
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--cont", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--mutant-cont", type=int, default=16)
    ap.add_argument("--ttft-lengths", default="512,2048,4096")
    ap.add_argument("--ttft-rounds", type=int, default=5)
    ap.add_argument("--stand-in-attention", action="store_true", help="REHEARSAL: SDPA over the pool's fp8 bytes")
    ap.add_argument("--placement", default="all-vram", help="REHEARSAL when not all-vram")
    ap.add_argument("--vram-gb", default="1.2")
    a = ap.parse_args()
    lengths = [int(x) for x in a.ttft_lengths.split(",")]
    need = max(max(lengths) + 1, a.prompt + a.cont + 2)
    os.environ.update({
        "E4B_PAGED_MODEL": a.model, "E4B_PAGED_REVISION": a.revision, "E4B_PAGED_ARENA": a.arena,
        "E4B_PAGED_CALIB": a.calib, "E4B_PAGED_MAX_SEQS": "4", "E4B_PAGED_GRAPHS": "0",
        "E4B_PAGED_PLACEMENT": a.placement, "E4B_PAGED_VRAM_GB": a.vram_gb,
        "E4B_PAGED_MAX_TOKENS_PER_SEQ": str(need), "E4B_PAGED_CHUNK_TOKENS": str(a.chunk)})

    import transformers
    import transformers.models.qwen3_5.modeling_qwen3_5 as m_dense
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as m_moe

    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

    toggle = GdnToggle([m_dense, m_moe])
    t0 = time.time()
    parts = build_engine(PagedServeConfig.from_env())
    build_s = time.time() - t0
    runner, sched, tok = parts.runner, parts.scheduler, parts.tokenizer
    model = runner.model
    gdn_modules = sorted({type(m).__module__ for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet")})
    kv = runner.kv
    if a.stand_in_attention:
        def stand_in(layer, q, slots=None, sm_scale=None, window=None, sinks=None, **_):
            outs = []
            for b, slot in enumerate(slots):
                kr, vr = kv.reference_kv(layer, slot)
                outs.append(torch.nn.functional.scaled_dot_product_attention(
                    q[b][None, :, None].float(), kr.permute(1, 0, 2)[None].float(), vr.permute(1, 0, 2)[None].float(),
                    scale=sm_scale, enable_gqa=True)[0, :, 0].to(q.dtype))
            return torch.stack(outs)

        kv.attention = stand_in
    rec = {"model": a.model, "revision": a.revision, "transformers": transformers.__version__,
           "build_s": round(build_s, 1), "gdn_modules": gdn_modules,
           "toggle_resolved": toggle.resolved_record(), "args": vars(a),
           "rehearsal": {"stand_in_attention": a.stand_in_attention, "placement": a.placement},
           "engine": {k: v for k, v in parts.info.items() if k in ("moe_layers", "experts", "top_k", "model_type", "kv",
                                                                  "graph_status", "grouping")}}
    toggle.use("torch")
    rec["toggle_torch"] = toggle.record()
    toggle.use("resolved")
    rec["loaded_commit"] = loaded_commit(a.model, a.revision)

    span = a.prompt + a.cont + 1
    windows = wikitext_prompts(tok, list(range(a.windows)), span)
    tap = LogitTap(model)
    t1 = time.time()
    try:
        per = []
        for w, window in enumerate(windows):
            paths = {"torch": "torch", "kernels": "resolved"}
            if w < a.null_windows:
                paths["torch_again"] = "torch"
            got = lockstep(runner, tap, toggle, window, prompt=a.prompt, cont=a.cont, chunk=a.chunk, paths=paths,
                           rid0=10_000 + 10 * w)
            per.append(got)
            k = got["kernels"]
            print(f"P106_WINDOW {w}: prefill {summarize(k['prefill'])} | decode {summarize(k['decode'])}", flush=True)
        q_s = time.time() - t1
        kern = {ph: merge([p["kernels"][ph] for p in per]) for ph in ("prefill", "decode")}
        null = {ph: merge([p["torch_again"][ph] for p in per if "torch_again" in p]) for ph in ("prefill", "decode")}
        rec["quality"] = {"windows": a.windows, "prompt": a.prompt, "cont": a.cont, "chunk": a.chunk, "seconds": round(q_s, 1),
                          "kernels": {ph: summarize(v) for ph, v in kern.items()},
                          "null": {ph: summarize(v) for ph, v in null.items()},
                          "per_window": [{ph: summarize(p["kernels"][ph]) for ph in ("prefill", "decode")} for p in per],
                          "rows": {"kernels": kern, "null": null}}
        t2 = time.time()
        mut = lockstep(runner, tap, toggle, windows[0][:a.prompt + a.mutant_cont + 1], prompt=a.prompt,
                       cont=a.mutant_cont, chunk=a.chunk, paths={"torch": "torch", "mutant": mutant}, rid0=20_000)
        rec["mutant"] = {"what": "fla's delta rules called with use_qk_l2norm_in_kernel=False", "cont": a.mutant_cont,
                         "seconds": round(time.time() - t2, 1),
                         "result": {ph: summarize(mut["mutant"][ph]) for ph in ("prefill", "decode")}}
    finally:
        tap.close()
    t3 = time.time()
    base = a.windows
    prompts_by_len = {n: wikitext_prompts(tok, list(range(base, base + 1 + a.ttft_rounds)), n) for n in lengths}
    rec["ttft"] = ttft(sched, toggle, prompts_by_len, a.ttft_rounds)
    rec["ttft_seconds"] = round(time.time() - t3, 1)
    toggle.use("resolved")
    rec["toggle_final"] = toggle.record()
    rec["gpu"] = torch.cuda.get_device_name(0)
    rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    open(a.out, "w").write(json.dumps(rec))
    q, m = rec["quality"]["kernels"], rec["mutant"]["result"]
    nl = rec["quality"]["null"]
    print(f"P106_BOX: build {build_s:.0f}s | quality prefill KL {q['prefill']['mean_kl']:.3e} agree "
          f"{q['prefill']['argmax_agree']:.4f} d_nll {q['prefill']['mean_d_nll']:+.4f} | decode KL {q['decode']['mean_kl']:.3e} "
          f"agree {q['decode']['argmax_agree']:.4f} d_nll {q['decode']['mean_d_nll']:+.4f} | null identical "
          f"{nl['prefill'].get('identical_rows')}/{nl['prefill'].get('n')} + {nl['decode'].get('identical_rows')}/"
          f"{nl['decode'].get('n')} | mutant KL {m['prefill']['mean_kl']:.3e}/{m['decode']['mean_kl']:.3e} | TTFT speedup "
          f"{ {k: v['speedup'] for k, v in rec['ttft'].items()} } | mem {rec['max_mem_gb']} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
