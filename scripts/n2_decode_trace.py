"""N2 Phase-1 trace collector (docs/N2_PHASE01_RECONSTRUCTION.md; O1 adapter-active pairing).

Two modes:
  --generate            : greedy-decode 16 fresh alpaca prompts (256 new tokens each) and
                          record the trace; also saves the full token sequences to
                          sequences.json so other configs can replay them.
  --replay SEQUENCES    : teacher-force the SAME token sequences through this config
                          (prefill the prompt, then feed the generated tokens one at a time
                          with KV cache) — identical token streams per O1, so routing is
                          comparable across base/adapter/precision.

Per decode token per MoE layer, records the routed top-k expert set and the router top-k
boundary margin (k-th minus (k+1)-th logit). Prefill rows are excluded (decode-time temporal
locality is the question). Output: trace.jsonl (one row per decode token) + result.json.

Examples:
    python scripts/n2_decode_trace.py --job-dir runs/jobs/n2trace_nf4_base --quant-type nf4 --generate
    python scripts/n2_decode_trace.py --job-dir runs/jobs/n2trace_nf4_adapter --quant-type nf4 \\
        --adapter .../nf4/adapter_best.pt --replay runs/jobs/n2trace_nf4_base/sequences.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def load_model(quant_type, adapter_path):
    import torch
    import experts4bit_qlora.train as train
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora

    torch.manual_seed(0)
    model, _ = load_moe_4bit_streaming(train.MODEL, "cuda", torch.bfloat16, 8, 16,
                                       offload=False, pin=True, quant_type=quant_type)
    model.to("cuda")
    add_attention_lora(model, 8, 16, torch.bfloat16)
    if adapter_path:
        sd = torch.load(adapter_path, map_location="cuda")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if (len(sd) - len(unexpected)) == 0 or unexpected:
            raise RuntimeError("adapter load mismatch")
    model.eval()
    return model


class TraceRecorder:
    """Hooks every ExpertsLoRA (routed set) and router gate (boundary margin). Single-row
    calls are decode tokens; multi-row calls are prefill and are skipped."""

    def __init__(self, model):
        from experts4bit_qlora.lora import ExpertsLoRA

        self.per_layer_sets = {}
        self.per_layer_margins = {}
        self.handles = []
        moe = [m for m in model.modules() if isinstance(m, ExpertsLoRA)]
        self.n_layers = len(moe)
        for li, mod in enumerate(moe):
            self.handles.append(mod.register_forward_pre_hook(self._expert_hook(li)))
        # OLMoE's router is an OlmoeTopKRouter (NOT nn.Linear) whose forward returns
        # (router_logits[.,E], router_scores[.,k], router_indices[.,k]); match by name and
        # take output[0] for the full-logit top-k boundary margin.
        routers = [m for n, m in model.named_modules() if n.endswith("mlp.gate")]
        for li, mod in enumerate(routers):
            self.handles.append(mod.register_forward_hook(self._margin_hook(li)))

    def _expert_hook(self, li):
        def hook(module, args):
            tki = args[1]
            if tki.shape[0] == 1:  # decode token, not prefill
                self.per_layer_sets.setdefault(li, []).append(
                    sorted(int(x) for x in tki.reshape(-1).tolist()))
        return hook

    def _margin_hook(self, li):
        def hook(module, args, output):
            logits = output[0] if isinstance(output, (tuple, list)) else output
            if logits.shape[0] == 1:  # decode token (not prefill)
                vals = logits.detach().float().sort(dim=-1, descending=True).values[0]
                k = 8 if vals.shape[-1] > 8 else vals.shape[-1] - 1
                self.per_layer_margins.setdefault(li, []).append(
                    round(float(vals[k - 1] - vals[k]), 5))
        return hook

    def drain(self, prompt_idx, out_file):
        """Write one jsonl row per decode token collected since the last drain."""
        n_tok = min((len(v) for v in self.per_layer_sets.values()), default=0)
        for t in range(n_tok):
            row = {"prompt": prompt_idx, "token": t,
                   "layers": {str(li): self.per_layer_sets[li][t] for li in self.per_layer_sets},
                   "margins": {str(li): self.per_layer_margins[li][t]
                               for li in self.per_layer_margins if t < len(self.per_layer_margins[li])}}
            out_file.write(json.dumps(row) + "\n")
        self.per_layer_sets = {}
        self.per_layer_margins = {}
        return n_tok

    def remove(self):
        for h in self.handles:
            h.remove()


def build_prompts(offset, n):
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split=f"train[{offset}:{offset + n}]")
    prompts = []
    for ex in ds:
        head = f"### Instruction:\n{ex['instruction']}\n\n"
        if ex.get("input"):
            head += f"### Input:\n{ex['input']}\n\n"
        prompts.append(head + "### Response:\n")
    return prompts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--quant-type", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--replay", default=None, help="sequences.json from a --generate run")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--prompt-offset", type=int, default=11088)
    args = ap.parse_args()
    if bool(args.generate) == bool(args.replay):
        raise SystemExit("exactly one of --generate / --replay is required")
    os.makedirs(args.job_dir, exist_ok=True)

    import torch
    import experts4bit_qlora.train as train
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(train.MODEL)
    model = load_model(args.quant_type, args.adapter)
    rec = TraceRecorder(model)
    trace_path = os.path.join(args.job_dir, "trace.jsonl")
    total = 0

    with torch.no_grad(), open(trace_path, "w") as tf:
        if args.generate:
            prompts = build_prompts(args.prompt_offset, args.n_prompts)
            sequences = []
            for pi, prompt in enumerate(prompts):
                ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
                out = model.generate(ids, max_new_tokens=args.tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id, use_cache=True)
                sequences.append({"prompt_len": int(ids.shape[1]),
                                  "tokens": out[0].tolist()})
                total += rec.drain(pi, tf)
                print(f"[gen] prompt {pi + 1}/{len(prompts)} -> {total} tokens traced", flush=True)
            with open(os.path.join(args.job_dir, "sequences.json"), "w") as sf:
                json.dump({"prompt_offset": args.prompt_offset, "n_prompts": args.n_prompts,
                           "sequences": sequences}, sf)
        else:
            seqs = json.load(open(args.replay))["sequences"]
            for pi, s in enumerate(seqs):
                full = torch.tensor([s["tokens"]], device="cuda")
                plen = s["prompt_len"]
                out = model(input_ids=full[:, :plen], use_cache=True)
                rec.per_layer_sets = {}
                rec.per_layer_margins = {}  # drop prefill partial rows defensively
                past = out.past_key_values
                for t in range(plen, full.shape[1]):
                    out = model(input_ids=full[:, t:t + 1], past_key_values=past, use_cache=True)
                    past = out.past_key_values
                total += rec.drain(pi, tf)
                print(f"[replay] prompt {pi + 1}/{len(seqs)} -> {total} tokens traced", flush=True)
    rec.remove()

    result = {
        "job_type": "n2_trace", "status": "pass",
        "storage_mode": args.quant_type, "adapter": args.adapter,
        "mode": "generate" if args.generate else "replay",
        "n_prompts": args.n_prompts, "decode_tokens_traced": total,
        "n_moe_layers": rec.n_layers,
        "torch_version": torch.__version__, "gpu_name": torch.cuda.get_device_name(0),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(os.path.join(args.job_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"traced {total} decode tokens -> {trace_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
