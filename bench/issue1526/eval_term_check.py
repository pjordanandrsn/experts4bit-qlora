"""experts4bit-qlora#1526: the training estimate against HO1's held-out Qwen3-30B-A3B rows, each in its recorded configuration.

usage: python bench/issue1526/eval_term_check.py

In sample (the registered bar, gated by tests/test_eval_phase_estimate.py): peak <= estimate <= 1.05 x peak on HO1's 22 post-hoc
primary 4096 x 1 packed rows (experts4bit-qlora 0.48.0). Out of sample (reported here, not gated): TC1's tc1-5090-147 and -150
P129 arms. The peak is each receipt's ``peak_vram_gb`` (``torch.cuda.max_memory_allocated``, recorded in GB to three decimals).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GiB = 1 << 30


def estimate(adapters, tokens, env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        from experts4bit_qlora import QLoRASetup, describe_moe, estimate_qlora_footprint

        kw = dict(adapter_dtype="fp32") if adapters == "fp32" else dict(adapter_dtype="bf16", attention_adapter_dtype="fp32")
        fp = estimate_qlora_footprint(describe_moe(os.path.join(HERE, "qwen3-30b-a3b")),
                                      QLoRASetup(r=16, alpha=16, attn_4bit=True, expert_kernel="grouped_nf4", **kw),
                                      tokens_per_microbatch=tokens, optimizer="adamw_8bit")
        return fp.device_bytes
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def rows():
    d = json.load(open(os.path.join(HERE, "heldout-4096-rows.json")))
    env048 = {k: v for k, v in d["config_048"].items() if k.startswith("E4B_")}
    for r in d["in_sample"]:
        peak = json.load(open(os.path.join(ROOT, r["receipt"])))["peak_vram_gb"] * 1e9
        yield "in", r, peak, estimate(r["adapters"], 4096, env048)
    oc = d["out_of_sample_config"]
    for r in d["out_of_sample"]:
        env = {"E4B_ABSMAX_DQ": oc["E4B_ABSMAX_DQ"], "E4B_TRAIN_FUSE_QKV": "1" if r["fuse_qkv"] else "0"}
        peak = json.load(open(os.path.join(ROOT, r["receipt"])))["peak_vram_gb"] * 1e9
        yield "out", r, peak, estimate(r["adapters"], oc["tokens_per_microbatch"], env)


if __name__ == "__main__":
    for E4B in ("E4B_CHUNKED_LM_LOSS", "E4B_CHUNKED_EVAL_LOSS"):
        os.environ.pop(E4B, None)
    print("| sample | receipt | adapters | peak GiB | estimate GiB | estimate / peak | bar |")
    print("|---|---|---|---|---|---|---|")
    ok = True
    for kind, r, peak, est in rows():
        bar = "" if kind == "out" else ("PASS" if peak <= est <= 1.05 * peak else "FAIL")
        ok &= bar != "FAIL"
        print(f"| {kind} | {r['receipt'].split('receipts/')[1]} | {r['adapters']} | {peak / GiB:.3f} | {est / GiB:.3f} | "
              f"{est / peak:.4f} | {bar} |")
    sys.exit(0 if ok else 1)
