"""The grouped_nf4 MoE-backward branch of the training estimate (experts4bit-qlora#1526).

In sample: the four RTX A2000 probe receipts under bench/issue1526/receipts are the points this branch was set against,
so the test pins estimate >= measured on exactly those points, not a held-out claim."""
import json
import os

import pytest

from experts4bit_qlora import QLoRASetup, describe_moe, estimate_qlora_footprint
from experts4bit_qlora import recipe

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "issue1526")
RECEIPTS = sorted(f for f in os.listdir(os.path.join(HERE, "receipts")) if f.endswith(".json"))


def _device(fp):
    return sum(i.bytes for i in fp.items if i.where == "device")


def test_mirrored_rules_match_the_installed_kernel():
    q = pytest.importorskip("nf4_qlora")
    assert recipe.GNF4_PAD_BYTES_LIMIT == q._PAD_BYTES_LIMIT
    assert recipe.GNF4_PAD_BUCKETS_MIN_ROWS == q._PAD_BUCKETS_AUTO_MIN_ROWS
    assert recipe.GNF4_PAD_BUCKET_RATIO == q._PAD_BUCKET_RATIO
    ladder = getattr(q, "_ladder_up", None)
    if ladder is not None:
        assert all(recipe._ladder_up(n) == ladder(n) for n in range(0, 5000))


@pytest.mark.parametrize("name", RECEIPTS)
def test_estimate_covers_the_in_sample_probe(name):
    r = json.load(open(os.path.join(HERE, "receipts", name)))
    topo = describe_moe(os.path.join(HERE, "q3red-L4"))
    setup = QLoRASetup(adapter_dtype="fp32" if r["adapters"] == "fp32" else "bf16", attn_4bit=True,
                       expert_kernel="grouped_nf4", r=16, alpha=16)
    est = _device(estimate_qlora_footprint(topo, setup, tokens_per_microbatch=r["tokens_per_microbatch"],
                                           optimizer="adamw_8bit"))
    measured = max(r["peak_allocated_bytes_per_step"][1:])          # steady steps, after the optimizer's first step
    assert est >= measured, (name, est, measured)


def test_the_branch_is_the_grouped_kernels_only():
    topo = describe_moe(os.path.join(HERE, "q3red-L4"))
    for kernel in ("reference", "grouped_nf4"):
        for train_experts in (True, False):
            s = QLoRASetup(expert_kernel=kernel, train_experts=train_experts, adapter_dtype="fp32")
            b, how = recipe._grouped_nf4_backward(topo, s, 4096, layer=1)
            assert (b > 0) == (kernel == "grouped_nf4" and train_experts), (kernel, train_experts)


def test_bucket_rule_switches_at_the_kernels_threshold():
    kw = dict(n_experts=128, top_k=8, hidden=2048, first_out=1536, intermediate=768, adapter_dtype="bf16")
    rows, how = recipe.grouped_nf4_padded_rows_bound(tokens=2048, **kw)       # 16,384 routed rows: buckets
    assert rows == 2 * 2048 * 8 and how.startswith("bucketed")
    rows, how = recipe.grouped_nf4_padded_rows_bound(tokens=1024, **kw)       # 8,192: one padded block
    assert how.startswith("single block") and rows <= 128 * 1024


R29 = os.path.join(HERE, "receipts-29370aaa")
STATIC_CENSUS = ("frozen expert stacks", "attention projections (NF4)", "fused q/k/v training projection",
                 "dense weights (bf16)", "expert LoRA adapters", "attention LoRA adapters", "optimizer state")


def test_held_logits_close_the_native_t1024_row():
    """Derived, not fitted: at T=1024 (stock loss, under the chunked-loss gate) the native probe's measured loss branch is the
    stock workspace, T x V x LOGITS_LOSS_BYTES, plus the bf16 logits the probe holds on `out` through the backward
    (probe_census.py: `out = model(...); (out.loss / accum).backward()`). The census, not the estimate, gives the held bytes."""
    r = json.load(open(os.path.join(R29, "L4-T1024-native.json")))
    topo = describe_moe(os.path.join(HERE, "q3red-L4"))
    T, V, H = r["tokens_per_microbatch"], topo.vocab_size, topo.hidden_size
    assert (T, V) == (1024, 151936)
    fp = estimate_qlora_footprint(topo, QLoRASetup(adapter_dtype="bf16", attn_4bit=True, expert_kernel="grouped_nf4",
                                                   r=16, alpha=16), tokens_per_microbatch=T, optimizer="adamw_8bit")
    items = {i.name: i.bytes for i in fp.items if i.where == "device"}
    assert items["logits held by the caller"] == T * V * 2
    c = r["census"]
    held = sum(c["items_bytes"].get(k, 0) for k in STATIC_CENSUS) + c["grad_bytes_from_params"] + c["unclassified_allocated"]
    peak = max(r["peak_allocated_bytes_per_step"][1:])
    branch = peak - held - topo.n_layers * T * H * 2              # minus the saved layer inputs
    derived = T * V * recipe.LOGITS_LOSS_BYTES + items["logits held by the caller"]
    assert abs(branch - derived) <= 0.005 * peak, (branch, derived)  # 7.75 MiB of 4,852.7 recorded (0.16 %)
    assert branch - T * V * recipe.LOGITS_LOSS_BYTES > 0.9 * T * V * 2  # without the term the row is short by ~one logits copy


def test_held_logits_only_on_the_stock_loss():
    topo = describe_moe(os.path.join(HERE, "q3red-L4"))
    s = QLoRASetup(adapter_dtype="bf16", attn_4bit=True, expert_kernel="grouped_nf4", r=16, alpha=16)
    names = {T: {i.name for i in estimate_qlora_footprint(topo, s, tokens_per_microbatch=T, optimizer="adamw_8bit").items}
             for T in (1024, 4096)}
    assert "logits held by the caller" in names[1024]          # T x V x 4 under the 1 GiB gate: stock loss
    assert "logits held by the caller" not in names[4096]      # chunked: no logits are returned
