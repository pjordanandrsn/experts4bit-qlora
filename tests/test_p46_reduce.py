"""P46's reducer: the path census proves the mechanism, speed is read against auto, parity gates every arm."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("p46_reduce", os.path.join(HERE, "..", "bench", "p46", "p46_reduce.py"))
p46 = importlib.util.module_from_spec(_spec)
sys.modules["p46_reduce"] = p46
_spec.loader.exec_module(p46)


def _cell(status, sps, final, losses, peak, loop, padded, gmm=0, n=20):
    return {"status": status, "s_per_step_median_11plus": sps, "eval_loss_final": final, "losses": losses, "peak_vram_gb": peak, "tokens_per_s": 100,
            "kernel_calls_all": [{"fused_grouped_lora": 768, "lora_path_loop": loop, "lora_path_padded": padded, "lora_path_grouped_mm": gmm} for _ in range(n)]}


def _write(d, name, obj):
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(obj, f)


def test_p46_holds_on_the_registered_pattern(tmp_path):
    d = str(tmp_path)
    base = [2.0 - 0.05 * i for i in range(20)]
    _write(d, "qwen3_e4b_reference_attn4", _cell("ok", 40.0, 0.90, base, 24.0, 0, 0))
    _write(d, "qwen3_e4b_fused_attn4", _cell("ok", 29.3, 0.903, [x + 0.01 for x in base], 24.5, 192, 0))
    _write(d, "qwen3_e4b_fused_attn4_pad", _cell("ok", 14.0, 0.91, [x + 0.02 for x in base], 30.0, 0, 192))
    _write(d, "qwen3_e4b_fused_attn4_gmm", {"status": "refused", "reason": "torch._grouped_mm refused on this device"})
    v = p46.reduce(d)["verdicts"]
    assert v["P1"]["verdict"] == "HOLDS" and v["P2"]["verdict"] == "HOLDS" and abs(v["P2"]["ratio_to_auto"] - 14.0 / 29.3) < 1e-4
    assert v["P3"]["verdict"].startswith("NOT_READ") and v["P4"]["padded"]["passes"] and v["P4"]["auto"]["passes"] and v["P5"]["verdict"] == "HOLDS"
    md = p46.render_md(p46.reduce(d))
    assert "loop=[192]" in md and "padded=[192]" in md


def test_p46_refutations_and_parity_gate(tmp_path):
    d = str(tmp_path)
    base = [2.0 - 0.05 * i for i in range(20)]
    _write(d, "qwen3_e4b_reference_attn4", _cell("ok", 40.0, 0.90, base, 24.0, 0, 0))
    _write(d, "qwen3_e4b_fused_attn4", _cell("ok", 29.3, 0.903, base, 24.5, 100, 92))          # padded ran on some steps -> P1 refuted
    _write(d, "qwen3_e4b_fused_attn4_pad", _cell("ok", 26.0, 1.00, [x + 0.3 for x in base], 36.0, 0, 192))   # slow, off-parity, memory
    v = p46.reduce(d)["verdicts"]
    assert v["P1"]["verdict"] == "REFUTED" and v["P2"]["verdict"] == "REFUTED" and not v["P4"]["padded"]["passes"] and v["P5"]["verdict"] == "REFUTED"
    assert v["P3"]["verdict"].startswith("NOT_READ") and v["P4"]["grouped_mm"] == {"verdict": "NOT_READ"}
