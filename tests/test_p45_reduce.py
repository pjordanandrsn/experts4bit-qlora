"""P45's reducer applies the registered thresholds and reads a missing receipt as NOT_READ."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("p45_reduce", os.path.join(HERE, "..", "bench", "p45", "p45_reduce.py"))
p45 = importlib.util.module_from_spec(_spec)
sys.modules["p45_reduce"] = p45
_spec.loader.exec_module(p45)


def _prof(busy, fam, events):
    return {"summary": {"profiled_steps": 3, "wall_ms_per_step": 29000, "device_busy_fraction": busy, "memcpy_fraction_of_device": 0.01,
                        "device_events_per_step": events, "cpu_ops_per_step": 900000, "cpu_self_fraction_of_wall": 0.7,
                        "cpu_self_by_family_fraction": fam, "device_by_family_fraction": {}},
            "top_cpu": [{"name": "aten::index_select", "family": "routing", "self_cpu_ms": 5000.0}],
            "top_device": [{"name": "fused_grouped_lora_kernel", "family": "fused_kernel", "self_device_ms": 800.0}]}


def _write(d, name, obj):
    with open(os.path.join(d, name), "w") as f:
        json.dump(obj, f)


def test_verdicts_hold_on_the_registered_pattern(tmp_path):
    d = str(tmp_path)
    os.makedirs(os.path.join(d, "logs"))
    _write(d, "qwen3_e4b_fused_attn4.json", {"status": "ok", "s_per_step": 29.3})
    _write(d, "qwen3_e4b_fused_attn4_profile.json", _prof(0.16, {"routing": 0.35, "autograd": 0.2, "fused_kernel": 0.1, "other": 0.35}, 45000))
    _write(d, "qwen3_unsloth_ckpt_unsloth.json", {"status": "ok", "s_per_step": 12.0})
    _write(d, "qwen3_unsloth_ckpt_unsloth_profile.json", _prof(0.55, {"matmul": 0.6}, 9000))
    with open(os.path.join(d, "logs", "dmon_qwen3_e4b_fused_attn4.txt"), "w") as f:
        f.write("#Time        gpu    sm   mem   enc   dec  rxpci  txpci\n#HH:MM:SS    Idx     %     %     %     %   MB/s   MB/s\n")
        for i in range(10):
            f.write(f"03:0{i%10}:00       0    16    40     0     0    300    200\n")
    o = p45.reduce(d)
    v = o["verdicts"]
    assert v["P1"]["verdict"] == "HOLDS" and v["P2"]["verdict"].startswith("HOLDS") and v["P3"]["verdict"] == "HOLDS"
    assert v["P4"]["verdict"] == "HOLDS" and abs(v["P4"]["ratio"] - 0.55 / 0.16) < 1e-9
    assert v["P5"]["verdict"] == "HOLDS" and abs(v["P5"]["pcie_mean_gbps"] - 0.5) < 1e-9
    md = p45.render_md(o)
    assert "aten::index_select" in md and "Verdicts" in md


def test_alternatives_and_missing(tmp_path):
    d = str(tmp_path)
    _write(d, "qwen3_e4b_fused_attn4.json", {"status": "ok", "s_per_step": 29.3})
    _write(d, "qwen3_e4b_fused_attn4_profile.json", _prof(0.7, {"fused_kernel": 0.6, "routing": 0.1}, 3000))
    v = p45.reduce(d)["verdicts"]
    assert v["P1"]["verdict"] == "REFUTED" and v["P2"]["verdict"] == "ALTERNATIVE: ['fused_kernel']" and v["P3"]["verdict"] == "REFUTED"
    assert v["P4"]["verdict"].startswith("NOT_READ") and v["P5"]["verdict"].startswith("NOT_READ")
    assert p45.reduce(str(tmp_path / "nowhere"))["verdicts"]["P1"]["verdict"].startswith("NOT_READ")
