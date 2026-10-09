"""The estimate (loggetta main's planner device lines, with this branch's experts4bit_qlora on PYTHONPATH) against the
four measured granite-3.1-3b-a800m allocated peaks on the RTX A2000 (grouped_nf4, resident, E4B_ABSMAX_DQ=0)."""
import os
from loggetta import Workload, describe_model
from loggetta.backends import experts4bit

SETUP = dict(quant_type="nf4", blocksize=64, r=8, alpha=16, adapter_dtype="bf16", train_experts=True, train_attention=True,
             attn_4bit=False, expert_residency="device", pin=True, expert_kernel="grouped_nf4", dgrad=True, keep_moe_layers=0)
POINTS = [("T=1024 stock (G2 replay)", 512, "0", 3131.5), ("T=1024 chunked, forced (P1)", 512, "1", 2984.2),
          ("T=6144 stock (S6)", 3072, "0", 6522.7), ("T=6144 auto, chunks (A6)", 3072, "auto", 4105.3)]
topo = describe_model("ibm-granite/granite-3.1-3b-a800m-instruct")
print("| point | measured allocated MiB | estimate MiB | estimate − measured | activation item detail |")
print("|---|---|---|---|---|")
for name, seq, env, measured in POINTS:
    os.environ["E4B_CHUNKED_LM_LOSS"] = env
    lines, _, _ = experts4bit.estimate(topo, dict(SETUP), Workload(seq_len=seq, micro_batch=2))
    est = sum(ln[2] for ln in lines if ln[1] == "device") / 2**20
    act = next(ln for ln in lines if ln[0] == "activations")
    extra = next((ln[2] for ln in lines if ln[0].startswith("grouped_nf4 MoE backward")), 0) / 2**20
    print(f"| {name} | {measured:.1f} | {est:.1f} | {est - measured:+.1f} | act {act[2] / 2**20:.1f} + gnf4 line {extra:.1f}: "
          f"{act[4][:90]} |")
