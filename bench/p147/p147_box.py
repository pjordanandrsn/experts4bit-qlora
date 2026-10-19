#!/usr/bin/env python3
"""p147_box.py -- P147 on the lab's RTX A2000 (bench/p147/PREREG-p147.md): every arm in one process, teacher-forced.

  1. the shards: each safetensors shard's sha256 against eba92302's LFS sha256 (``model_files.json``);
  2. the windows, from the staged loaders at their pinned bytes:
       check_wikitext / check_c4val1: P141's 24 + 24 (``p115_quality.LOADERS``, sha256 == P141's);
       W1 / W16: wikitext every 1024 tokens (``p125_box.wikitext_windows``), the first 196 / 123;
       C4: c4val1, 16 windows;
  3. the model: e4b's streaming loader, experts NF4 offloaded to pinned host RAM, attention and head bf16 resident;
  4. the arms: A, A2 (A again), then -- only if instrument checks (a) and (b) hold -- I8ATTN, NF4ATTN, I8HEAD. Before
     every arm after A, every weight any arm touches is restored from a host copy and its sha256 checked.

Each window is 512 prompt tokens + 128 scored positions. A pass feeds tokens[:-1] and keeps the last 128 positions'
logits: the true token's NLL (fp32 log-softmax, the window's mean) and each position's argmax. W1 reads one window per
pass; every other set reads 16 per pass. No timing is recorded.

    python p147_box.py --model DIR --lock model_files.json --ref quality_off_mixtral.json --out OUT [--shakedown]
    python p147_box.py --self-test

``--shakedown`` (a dry id, before the registration merges) runs steps 1-3 and checks the check set's sha256 against
P141's, then ONE W1-shaped and ONE W16-shaped pass on synthetic token ids, recording peak GPU memory. It scores no real
window, so it reads no quality data.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPT, CONT = 512, 128
SETS = {   # set -> (loader, windows, group, stride)
    "check_wikitext": ("p97", 24, 1, None), "check_c4val1": ("c4", 24, 1, None),
    "W1": ("p125", 196, 1, 1024), "W16": ("p125", 123, 16, 1024), "C4": ("c4", 16, 16, None),
}
ARMS = (("A", None, None), ("A2", None, None), ("I8ATTN", "int8", "attn"), ("NF4ATTN", "nf4", "attn"),
        ("I8HEAD", "int8", "head"))
EXIT_SHARDS, EXIT_CHECKS = 13, 12


def windows_sha(ws) -> str:
    return hashlib.sha256(json.dumps(ws).encode()).hexdigest()


def build_windows(tok) -> dict:
    import p125_box
    import p115_quality as q
    out = {}
    for name, (loader, n, group, stride) in SETS.items():
        if loader == "p97":
            ws = q.LOADERS["wikitext"](tok, n, PROMPT, CONT)
        elif loader == "c4":
            ws = q.LOADERS["c4val1"](tok, n, PROMPT, CONT)
        else:
            ws = p125_box.wikitext_windows(tok, n, PROMPT, CONT, stride=stride)
        out[name] = {"windows": ws, "group": group, "stride": stride, "sha256": windows_sha(ws)}
    return out


def shard_check(model_dir, lock) -> dict:
    """Every shard's sha256 against the lock (eba92302's LFS sha256s), plus config.json's and the index's git blob ids."""
    bad = []
    for f in lock["shards"]:
        h = hashlib.sha256()
        with open(os.path.join(model_dir, f["file"]), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != f["sha256"]:
            bad.append(f["file"])
    for f in lock["small"]:
        data = open(os.path.join(model_dir, f["file"]), "rb").read()
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if blob != f["blob_id"]:
            bad.append(f["file"])
    return {"shards_ok": not bad, "mismatched": bad, "checked": len(lock["shards"]) + len(lock["small"])}


def score(model, windows, group, device) -> list:
    """Per window: the mean NLL over the 128 scored positions and their argmax, teacher-forced in passes of ``group``."""
    import torch
    out = []
    for i in range(0, len(windows), group):
        batch = torch.tensor([w[:-1] for w in windows[i:i + group]], device=device)
        target = torch.tensor([w[PROMPT:] for w in windows[i:i + group]], device=device)
        with torch.no_grad():
            logits = model(input_ids=batch, logits_to_keep=CONT, use_cache=False).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        am = logits.argmax(dim=-1)
        for k in range(batch.shape[0]):
            out.append({"nll": float(nll[k].mean()), "argmax": am[k].tolist()})
        del logits, logp
    return out


def score_arm(model, wins, device) -> dict:
    return {name: {"windows": len(w["windows"]), "group": w["group"], "sha256": w["sha256"],
                   **({"stride": w["stride"]} if w["stride"] else {}),
                   "per_window": score(model, w["windows"], w["group"], device)} for name, w in wins.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model")
    ap.add_argument("--lock")
    ap.add_argument("--ref")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--shakedown", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    import torch
    import p147_fakequant as fq
    import p147_reduce as red
    os.makedirs(a.out, exist_ok=True)
    put = lambda name, obj: json.dump(obj, open(os.path.join(a.out, name), "w"))   # noqa: E731
    run = {"lane": "P147", "model_dir": a.model, "versions": {"torch": torch.__version__}}
    import transformers
    import experts4bit_qlora as e4b
    import bitsandbytes
    run["versions"].update(transformers=transformers.__version__, e4b=getattr(e4b, "__version__", "?"),
                           bitsandbytes=bitsandbytes.__version__)
    import importlib.metadata as md
    run["versions"]["gnf4"] = md.version("grouped-nf4-gemm")
    run["card"] = torch.cuda.get_device_name(0)
    run.update(shard_check(a.model, json.load(open(a.lock))))
    put("run.json", run)
    print("P147_SHARDS " + json.dumps({k: run[k] for k in ("shards_ok", "mismatched", "checked")}), flush=True)
    if not run["shards_ok"]:
        return EXIT_SHARDS
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    wins = build_windows(tok)
    run["windows"] = {k: {"windows": len(v["windows"]), "sha256": v["sha256"]} for k, v in wins.items()}
    put("run.json", run)
    model, _ = e4b.load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, 16, 16, offload=True, pin=True,
                                            prefetch=True, quant_type="nf4")
    model.eval()
    ref = json.load(open(a.ref))
    if a.shakedown:
        want = {s: ref["windows_sha256"][t] for s, t in (("check_wikitext", "wikitext"), ("check_c4val1", "c4val1"))}
        got = {s: wins[s]["sha256"] for s in want}
        g = torch.Generator().manual_seed(0)
        shapes = {}
        for name, group in (("W1", 1), ("W16", 16)):
            torch.cuda.reset_peak_memory_stats()
            synth = [torch.randint(0, 32000, (PROMPT + CONT,), generator=g).tolist() for _ in range(group)]
            per = score(model, synth, group, "cuda")
            shapes[name] = {"windows": len(per), "argmax_positions": len(per[0]["argmax"]),
                            "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        census = {s: len(fq.targets(model, s)) for s in ("attn", "head")}
        sk = {"check_sha_match": got == want, "windows": run["windows"], "synthetic_passes": shapes, "targets": census}
        put("shakedown.json", sk)
        print("P147_SHAKEDOWN " + json.dumps(sk), flush=True)
        return 0 if got == want and census == {"attn": 128, "head": 1} else 3
    snap = fq.snapshot(model)
    arms = {}
    for name, fmt, scope in ARMS:
        rec = {"arm": name}
        if name != "A":
            rec["restore"] = fq.restore(model, snap)
        if fmt:
            census = fq.apply(model, fmt, scope)
            rec["census"] = {k: census[k] for k in ("format", "scope", "modules", "lm_head", "names", "rel_fro_err_max")}
        rec["sets"] = score_arm(model, wins, "cuda")
        put(f"arm_{name}.json", rec)
        arms[name] = rec
        print(f"P147_ARM {name} " + json.dumps({s: round(sum(w['nll'] for w in v['per_window']) / len(v['per_window']), 5)
                                                for s, v in rec["sets"].items()}), flush=True)
        if name == "A2":
            out, why = red.checks(arms, ref)
            put("checks.json", {"checks": out, "why": why})
            print("P147_CHECKS " + json.dumps({"why": why}), flush=True)
            if why:
                return EXIT_CHECKS
    return 0


def self_test() -> int:
    """The scoring on a tiny CPU causal LM with a fixed seed: shapes, the 128-position slice, NLL against a manual
    computation, grouped and ungrouped passes equal, and the window hash."""
    import torch
    cases = []

    class Tiny(torch.nn.Module):
        def __init__(self, v=50, h=16):
            super().__init__()
            torch.manual_seed(0)
            self.emb, self.out = torch.nn.Embedding(v, h), torch.nn.Linear(h, v)

        def forward(self, input_ids, logits_to_keep=0, use_cache=False):
            x = self.emb(input_ids).cumsum(1)
            lg = self.out(x)
            return type("O", (), {"logits": lg[:, -logits_to_keep:] if logits_to_keep else lg})

    m = Tiny()
    g = torch.Generator().manual_seed(1)
    ws = [torch.randint(0, 50, (PROMPT + CONT,), generator=g).tolist() for _ in range(5)]
    one, grouped = score(m, ws, 1, "cpu"), score(m, ws, 4, "cpu")
    cases.append(("five windows scored, 128 argmax positions each", len(one) == 5 and all(len(w["argmax"]) == CONT for w in one), None))
    cases.append(("grouped and one-per-pass agree", all(abs(x["nll"] - y["nll"]) < 1e-5 and x["argmax"] == y["argmax"]
                                                        for x, y in zip(one, grouped)), None))
    full = m(torch.tensor([ws[0][:-1]])).logits.float()[0]
    lp = torch.log_softmax(full, -1)
    manual = float(-sum(lp[PROMPT - 1 + j, ws[0][PROMPT + j]] for j in range(CONT)) / CONT)
    cases.append(("the NLL is the mean over positions 512..639 predicted from 511..638", abs(manual - one[0]["nll"]) < 1e-5,
                  (manual, one[0]["nll"])))
    cases.append(("the window hash is P115's json sha256", windows_sha([[1, 2]]) == hashlib.sha256(b"[[1, 2]]").hexdigest(), None))
    cases.append(("five sets, W1 one per pass, the rest sixteen", set(SETS) == {"check_wikitext", "check_c4val1", "W1", "W16", "C4"}
                  and SETS["W1"][2] == 1 and all(SETS[s][2] == 16 for s in ("W16", "C4")), None))
    bad = [(n, d) for n, ok, d in cases if not ok]
    print(f"p147_box self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
