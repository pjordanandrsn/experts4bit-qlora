"""The held-out evaluation in the training estimate, and attention adapters in their own dtype (experts4bit-qlora#1526).

In sample: HO1's 22 post-hoc primary 4096 x 1 packed Qwen3-30B-A3B rows (bench/issue1526/heldout-4096-rows.json, each
receipt under bench/h2h-2026-10-02/tc1/receipts), priced in the configuration they recorded (experts4bit-qlora 0.48.0).
The registered bar: peak <= estimate <= 1.05 x peak on every row. tc1-5090-147 and -150 are out of sample and only reported,
by bench/issue1526/eval_term_check.py. The rows pass because the evaluation term dominates them; this does not certify the
training-step term at 4,096 tokens on main, which no receipt here measures."""
import json
import os

import pytest

from experts4bit_qlora import QLoRASetup, describe_moe, estimate_qlora_footprint
from experts4bit_qlora import recipe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "bench", "issue1526")
ROWS = json.load(open(os.path.join(HERE, "heldout-4096-rows.json")))
CENSUS_EVAL = os.path.join(ROOT, "bench/h2h-2026-10-02/tc1/receipts/tc1-5090-112/qwen3memc4kb_e4b_fused_attn4_m_p4d.json")
QWEN3 = os.path.join(HERE, "qwen3-30b-a3b")
MiB = 1 << 20


def _setup(adapters):
    kw = dict(adapter_dtype="fp32") if adapters == "fp32" else dict(adapter_dtype="bf16", attention_adapter_dtype="fp32")
    return QLoRASetup(r=16, alpha=16, attn_4bit=True, expert_kernel="grouped_nf4", **kw)


@pytest.fixture
def config_048(monkeypatch):
    for k, v in ROWS["config_048"].items():
        if k.startswith("E4B_"):
            monkeypatch.setenv(k, v)
    monkeypatch.delenv("E4B_CHUNKED_LM_LOSS", raising=False)       # recorded: unset, the training loss chunked under auto
    return describe_moe(QWEN3)


@pytest.mark.parametrize("row", ROWS["in_sample"], ids=lambda r: r["receipt"].split("receipts/")[1])
def test_the_estimate_covers_each_in_sample_row_within_the_registered_bar(config_048, row):
    peak = json.load(open(os.path.join(ROOT, row["receipt"])))["peak_vram_gb"] * 1e9
    est = estimate_qlora_footprint(config_048, _setup(row["adapters"]), tokens_per_microbatch=4096,
                                   optimizer="adamw_8bit").device_bytes
    assert peak <= est <= 1.05 * peak, (row["receipt"], est, peak)


def test_the_evaluation_term_closes_on_the_census_at_its_peak(config_048):
    """tc1-5090-112's allocator census at its s20.eval peak: everything but gradients, plus the stock loss's buffers."""
    census = json.load(open(CENSUS_EVAL))["mem_census"]
    assert census["peak_window"]["peak_phase"] == "s20.eval"
    top = census["live_at_peak_top"]
    held = sum(g["bytes"] for g in top if g["group"].startswith("static:"))
    loss = sum(g["bytes"] for g in top if g["group"].startswith("site:"))
    fp = estimate_qlora_footprint(config_048, _setup("fp32"), tokens_per_microbatch=4096, optimizer="adamw_8bit")
    est_held = sum(i.bytes for i in fp.items if i.where == "device"
                   and i.name not in ("adapter gradients", "activations", "held-out evaluation, above the training step",
                                      "small allocations"))
    T, V, H = 4096, config_048.vocab_size, config_048.hidden_size
    assert abs(est_held - held) < MiB, (est_held, held)
    assert abs(T * V * recipe.EVAL_STOCK_BYTES_PER_LOGIT + T * H * recipe.EVAL_HIDDEN_BYTES - loss) < MiB
    assert fp.device_bytes >= census["peak_allocated_bytes"]


def test_the_device_total_is_the_larger_of_the_training_step_and_the_evaluation(config_048, monkeypatch):
    def items(eval_switch):
        monkeypatch.setenv("E4B_CHUNKED_EVAL_LOSS", eval_switch)
        return estimate_qlora_footprint(config_048, _setup("fp32"), tokens_per_microbatch=4096, optimizer="adamw_8bit")

    stock, chunked = items("0"), items("1")
    ev = next(i for i in stock.items if i.name.startswith("held-out"))
    assert ev.bytes > 0 and ev.basis == "heuristic"
    step = stock.device_bytes - ev.bytes
    assert chunked.device_bytes >= step                              # the item is never negative
    assert stock.device_bytes > chunked.device_bytes                 # stock evaluation holds the whole fp32 logits twice


def test_the_stock_evaluation_is_priced_wherever_the_model_is_not_patched(config_048, monkeypatch):
    monkeypatch.setenv("E4B_CHUNKED_EVAL_LOSS", "1")
    patched = estimate_qlora_footprint(config_048, _setup("fp32"), tokens_per_microbatch=4096, optimizer="adamw_8bit")
    monkeypatch.setenv("E4B_CHUNKED_LM_LOSS", "0")                  # no patch: evaluation is stock whatever the eval switch says
    bare = estimate_qlora_footprint(config_048, _setup("fp32"), tokens_per_microbatch=4096, optimizer="adamw_8bit")
    detail = next(i.detail for i in bare.items if i.name.startswith("held-out"))
    assert "E4B_CHUNKED_LM_LOSS=0" in detail and bare.device_bytes > patched.device_bytes


def test_attention_adapters_take_their_own_dtype(config_048):
    native = estimate_qlora_footprint(config_048, _setup("native"), tokens_per_microbatch=4096, optimizer="adamw_8bit")
    allbf16 = estimate_qlora_footprint(config_048, QLoRASetup(r=16, alpha=16, attn_4bit=True, adapter_dtype="bf16"),
                                       tokens_per_microbatch=4096, optimizer="adamw_8bit")
    n = 16 * config_048.attention.in_plus_out

    def item(fp, name):
        return next(i.bytes for i in fp.items if i.name == name)

    assert item(native, "attention LoRA adapters") == 4 * n and item(allbf16, "attention LoRA adapters") == 2 * n
    assert item(native, "expert LoRA adapters") == item(allbf16, "expert LoRA adapters")
    assert item(native, "adapter gradients") - item(allbf16, "adapter gradients") == 2 * n
    explicit = QLoRASetup(r=16, alpha=16, attn_4bit=True, adapter_dtype="bf16", attention_adapter_dtype="bf16")
    assert estimate_qlora_footprint(config_048, explicit, tokens_per_microbatch=4096,
                                    optimizer="adamw_8bit").device_bytes == allbf16.device_bytes   # None = adapter_dtype


def test_an_unknown_attention_adapter_dtype_is_refused(config_048):
    fp = estimate_qlora_footprint(config_048, QLoRASetup(attention_adapter_dtype="fp16"), tokens_per_microbatch=512)
    assert fp.items == () and any("attention_adapter_dtype" in r for r in fp.refusals)
