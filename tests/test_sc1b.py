# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""CPU tests for lane SC1b (#846): the census reducer's self-test, the e4b census bracket, and the frozen class map v0
applied to one synthetic decode layer per engine (kernel names from grouped-nf4-gemm 34da93d and the bench/sc1b upstream
notes). The layer sequences state, before any capture, what the map is meant to do; a capture that disagrees reads as
CLASS_MAP_INCOMPLETE or as a listed map delta, never as a silent re-bin."""
from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

LANE = Path(__file__).resolve().parents[1] / "bench" / "sc1b"


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, LANE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


census = _mod("sc1b_census")


def _k(name, grid=(1, 1, 1)):
    return {"name": name, "demangled": name, "grid": grid, "kind": "kernel"}


def _classes(engine, seq, in_graph=True):
    spec = census.load_classes(str(LANE / "kernel_classes.json"), engine)
    return census.classify_seq(seq, spec, in_graph)


def test_reducer_self_test():
    out = subprocess.run([sys.executable, str(LANE / "sc1b_census.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_e4b_census_bracket_selftest():
    out = subprocess.run([sys.executable, str(LANE / "sc1b_e4b_census.py"), "--selftest"], capture_output=True, text=True)
    assert out.returncode == 0 and "selftest OK" in out.stdout, out.stdout + out.stderr


def test_class_map_is_valid_for_every_engine():
    cmap = json.loads((LANE / "kernel_classes.json").read_text())
    for eng in ("e4b", "vllm", "sglang", "llamacpp"):
        spec = census.load_classes(str(LANE / "kernel_classes.json"), eng)
        assert spec["segment"]["start"] and spec["segment"]["end"] and spec["expert_names"], eng
        assert eng in cmap


def test_e4b_layer_b1():
    # one B=1 decode layer: int4 attention projections (gridY = R = 1) share _quant_x_rows / _gemv_int4_b32 /
    # _reduce_partials with the experts (gridY = top_k = 8); the cuBLAS router GEMM precedes the router epilogue
    seq = [_k("_rmsnorm_rows"), _k("_quant_x_rows"), _k("_gemv_int4_b32", (64, 1, 2)), _k("_reduce_partials"),
           _k("_rope_norm_heads"), _k("_fp8_append_t1_side"), _k("_fp8_paged_decode_split"), _k("_fp8_combine"),
           _k("_quant_x_rows"), _k("_gemv_int4_b32", (32, 1, 2)), _k("_reduce_partials"), _k("_rmsnorm_resid_rows"),
           _k("nvjet_hsh_64x8"), _k("_router_epilogue"), _k("_quant_x_rows_gathered"), _k("_gemv_int4_b32", (24, 8, 2)),
           _k("_reduce_partials"), _k("_swiglu_rows"), _k("_quant_x_rows_gathered"), _k("_gemv_int4_b32", (32, 8, 2)),
           _k("_reduce_partials"), _k("_combine_rows"), _k("_scaled_resid_add_rows")]
    want = ["norm_elem", "dense_gemm", "dense_gemm", "dense_gemm", "norm_elem", "attn", "attn", "attn",
            "dense_gemm", "dense_gemm", "dense_gemm", "norm_elem", "dense_gemm", "moe_route", "moe_expert", "moe_expert",
            "moe_expert", "moe_expert", "moe_expert", "moe_expert", "moe_expert", "moe_route", "norm_elem"]
    assert _classes("e4b", seq) == want, list(zip([s["name"] for s in seq], _classes("e4b", seq), want))


def test_vllm_layer_and_out_of_graph():
    seq = [_k("triton_red_fused_add_rms_norm_0"), _k("void marlin::Marlin<half>"), _k("triton_poi_fused_rotary_1"),
           _k("reshape_and_cache_flash_kernel"), _k("flash_fwd_splitkv_kernel"), _k("void marlin::Marlin<half>"),
           _k("triton_red_fused_add_rms_norm_2"), _k("nvjet_hsh_128x64"), _k("topkGating<float, 128>"),
           _k("moe_align_block_size_kernel"), _k("marlin_moe_wna16::Marlin<half>"), _k("act_and_mul_kernel"),
           _k("marlin_moe_wna16::Marlin<half>"), _k("moe_sum_kernel")]
    want = ["norm_elem", "dense_gemm", "norm_elem", "attn", "attn", "dense_gemm", "norm_elem", "dense_gemm", "moe_route",
            "moe_route", "moe_expert", "moe_expert", "moe_expert", "moe_route"]
    assert _classes("vllm", seq) == want, list(zip([s["name"] for s in seq], _classes("vllm", seq), want))
    out = [_k("_prepare_pos_seq_lens_kernel"), _k("_compute_slot_mappings_kernel"), _k("nvjet_hsh_256x128"),
           _k("_gumbel_sample_kernel"), _k("vectorized_elementwise_kernel")]
    assert _classes("vllm", out, in_graph=False) == ["input_prep", "input_prep", "dense_gemm", "sample", "input_prep"]


def test_sglang_layer():
    seq = [_k("FusedAddRMSNormKernel"), _k("sglang::device::marlin::Marlin<half>"), _k("fused_qknorm_warp"), _k("fused_rope_kernel"),
           _k("store_kvcache"), _k("BatchPrefillWithPagedKVCacheKernel"), _k("sglang::device::marlin::Marlin<half>"),
           _k("FusedAddRMSNormKernel"), _k("nvjet_hsh_128x64"), _k("_router_triton_kernel"), _k("align_single_token_kernel"),
           _k("vectorized_elementwise_kernel<FillFunctor>"), _k("sglang::device::marlin_moe::Marlin<half>"), _k("act_and_mul_kernel"),
           _k("sglang::device::marlin_moe::Marlin<half>"), _k("moe_sum_reduce_kernel")]
    want = ["norm_elem", "dense_gemm", "norm_elem", "norm_elem", "attn", "attn", "dense_gemm", "norm_elem", "dense_gemm",
            "moe_route", "moe_route", "moe_route", "moe_expert", "moe_expert", "moe_expert", "moe_route"]
    assert _classes("sglang", seq) == want, list(zip([s["name"] for s in seq], _classes("sglang", seq), want))


def test_llamacpp_b16_stream_k_is_placed_by_segment_not_grid():
    # at 16 tokens dense and expert MMQ both launch with gridY = gridZ = 1 (stream-k): only node order separates them;
    # quantize_mmq_q8_1 inherits the class of the matmul it feeds; the stream-k fixup inherits from the matmul before it
    seq = [_k("rms_norm_f32"), _k("quantize_mmq_q8_1"), _k("mul_mat_q<12>", (170, 1, 1)), _k("mul_mat_q_stream_k_fixup"),
           _k("rms_norm_mul_rope_f32"), _k("flash_attn_ext_f16"), _k("flash_attn_combine_results"),
           _k("quantize_mmq_q8_1"), _k("mul_mat_q<12>", (170, 1, 1)), _k("mul_mat_q_stream_k_fixup"), _k("k_bin_bcast"),
           _k("mul_mat_f"), _k("topk_moe_cuda"), _k("mm_ids_helper"), _k("quantize_mmq_q8_1"), _k("mul_mat_q<12>", (170, 1, 1)),
           _k("mul_mat_q_stream_k_fixup"), _k("unary_gated_op_kernel"), _k("quantize_mmq_q8_1"), _k("mul_mat_q<14>", (170, 1, 1)),
           _k("mul_mat_q_stream_k_fixup"), _k("moe_weighted_reduction_f32")]
    want = ["norm_elem", "dense_gemm", "dense_gemm", "dense_gemm", "norm_elem", "attn", "attn", "dense_gemm", "dense_gemm",
            "dense_gemm", "norm_elem", "dense_gemm", "moe_route", "moe_route", "moe_expert", "moe_expert", "moe_expert",
            "moe_expert", "moe_expert", "moe_expert", "moe_expert", "moe_route"]
    assert _classes("llamacpp", seq) == want, list(zip([s["name"] for s in seq], _classes("llamacpp", seq), want))


def test_an_unknown_kernel_is_residual_never_a_guess():
    for eng in ("e4b", "vllm", "sglang", "llamacpp"):
        assert _classes(eng, [_k("some_new_fused_kernel_v2")]) == ["residual"], eng


# ---- box D (the census box) wired into SC1's scripts --------------------------------------------------------------------
SC1 = LANE.parent / "sc1"


def test_box_d_is_wired_into_sc1s_box_script_and_controller():
    run = (SC1 / "sc1_run.sh").read_text()
    assert 'case "$BOX" in A|B|C|D) ;;' in run and 'C|D) BASEPY=python3;;' in run
    assert "D) . $W/sc1b_box_d.sh; install_vllm; install_sglang; install_llamacpp; install_nsys ;;" in run
    assert 'D) PROVE_NEEDS="vllm sglang llamacpp nsys";;' in run and '[ "$BOX" = D ] && prove_d' in run
    assert 'C) box_c;; D) box_d;; esac' in run
    drive = (SC1 / "sc1_drive.sh").read_text()
    assert 'case "$SC1_BOX" in A|B|C|D) ;;' in drive and 'sc1b_*|kernel_classes.json) src="$SC1B/$name"' in drive


def test_every_nsys_call_in_box_d_uses_the_pinned_binary():
    # the CUDA 13 image puts Nsight Compute's own nsys (2025.3.1) on PATH; box D must only ever run the pinned 2025.6.1
    import re
    for ln in (LANE / "sc1b_box_d.sh").read_text().splitlines():
        code = ln.split("#", 1)[0]
        assert not re.search(r'(^|[;&|(]\s*|\bthen\s+|\bdo\s+)nsys\s', code), ln
    serve = (LANE / "sc1b_serve_census.py").read_text()
    assert "subprocess" not in serve and '"nsys"' not in serve               # the client never talks to nsys (round 2 M1)
    box = (LANE / "sc1b_box_d.sh").read_text()
    assert " start --session" not in box and " launch --session" not in box   # 2025.6.1 documents neither as used before


def test_the_launch_prefix_is_empty_unless_set():
    # SC1's server commands are byte-identical when SC1_LAUNCH_PREFIX is unset (set -u, as the box script runs)
    snippet = 'set -u; prefix=(${SC1_LAUNCH_PREFIX:-}); echo ${prefix[@]+"${prefix[@]}"} cmd'
    assert subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}).stdout.strip() == "cmd"
    out = subprocess.run(["bash", "-c", snippet], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "SC1_LAUNCH_PREFIX": "/opt/nsys launch --session-new=s"}).stdout.strip()
    assert out == "/opt/nsys launch --session-new=s cmd"
    assert 'nohup ${prefix[@]+"${prefix[@]}"} "$LLAMACPP_BIN/llama-server"' in (SC1 / "llamacpp" / "llamacpp_box.sh").read_text()
    assert 'setsid ${PREFIX[@]+"${PREFIX[@]}"} "${CMD[@]}"' in (SC1 / "sglang" / "server.sh").read_text()


@pytest.mark.skipif(not Path("/proc/self/comm").exists(), reason="end_capture reads /proc/PID/comm (Linux)")
def test_end_capture_signals_the_app_and_never_nsys(tmp_path):
    # nsys's command line carries the app's, and so may the caller's: here the calling `bash -c` script holds the pattern too.
    # Only the app (a child of nsys) may be signalled; nsys must exit on its own once the app is gone.
    import os
    import signal
    box = (LANE / "sc1b_box_d.sh").read_text()
    fn = box[box.index("_descendants(){"):]
    fn = fn[:fn.index("return 1; fi; }") + len("return 1; fi; }")]
    d = tmp_path / "marker-sc1b-ec"
    d.mkdir()
    (d / "app").write_bytes(Path("/bin/sleep").read_bytes())
    (d / "app").chmod(0o755)
    (d / "nsys").write_text('#!/bin/bash\nd=$(dirname "$0"); "$d/app" 300 & echo $! > "$d/app.pid"; wait $!; echo "app=$?" > "$d/app.status"; exit 0\n')
    (d / "nsys").chmod(0o755)
    script = f"""line(){{ echo "$@"; }}
{fn}
{d}/nsys > /dev/null 2>&1 & N=$!
for i in $(seq 1 50); do [ -s {d}/app.pid ] && break; sleep 0.1; done
end_capture $N marker-sc1b-ec; echo "ec=$?"; wait $N; echo "nsys=$?"; cat {d}/app.status
"""
    try:
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30).stdout.split()
    finally:
        pid = (d / "app.pid").read_text().strip() if (d / "app.pid").exists() else ""
        if pid:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    assert out == ["ec=0", "nsys=0", "app=143"], out


def test_the_registered_window_is_one_set_of_numbers():
    box = (LANE / "sc1b_box_d.sh").read_text()
    assert "SC1B_SKIP=32; SC1B_STEPS=64; SC1B_TOKENS=160" in box
    vl, sv = _mod("sc1b_vllm_census"), _mod("sc1b_serve_census")
    assert (vl.SKIP, vl.STEPS, vl.TOKENS) == (sv.SKIP, sv.STEPS, sv.TOKENS) == (32, 64, 160)
    assert vl.profiler_config(32, 64)["delay_iterations"] == 35                 # decode steps 34-97, as e4b's bracket
    assert 'ap.add_argument("--skip", type=int, default=32)' in (LANE / "sc1b_e4b_census.py").read_text()


def test_the_prefill_attention_route_is_pinned_like_a13():
    run = (SC1 / "sc1_run.sh").read_text()
    pin = "unset E4B_PAGED_PREFILL_ATTN; export E4B_PAGED_PREFILL_ATTN=math"
    assert run.index("export E4B_INT4_PREFILL=loop") < run.index(pin) < run.index("<<'PYT'")


def test_the_toy_check_reads_the_census_records(tmp_path):
    toy = _mod("sc1b_toy")
    g, n, bad = tmp_path / "g.sqlite", tmp_path / "n.sqlite", tmp_path / "bad.sqlite"
    eager = [(1000 * i + 400, 1000 * i + 450, 70 + i, None, "elementwise_mul", (2, 1, 1)) for i in range(20)]
    h2d = [(1000 * i + 500, 1000 * i + 520, 90 + i, None, 4 << 16, 1) for i in range(20)]
    census._mkdb(str(g), kernels=eager, copies=h2d, graphs=[(1000 * i, 1000 * i + 300, 50 + i, 7) for i in range(20)])
    ks, ls = [], []
    for i in range(20):
        t, c = 100_000 * (i + 1), 900 + i
        ls.append((t - 5, t - 1, c))
        ks += [(t + 10 * j, t + 10 * j + 8, c, j + 1, f"k{j}", (4, 1, 1)) for j in range(3)]
        ks.append((t + 50, t + 60, 2000 + i, 0, "elementwise_mul", (2, 1, 1)))          # an export writing 0 for "no node"
    nh2d = [(100_000 * (i + 1) + 70, 100_000 * (i + 1) + 80, 3000 + i, None, 4 << 16, 1) for i in range(20)]
    census._mkdb(str(n), kernels=ks, copies=nh2d, launches=ls)
    assert toy.check(str(g), str(n)) == 0
    census._mkdb(str(bad), kernels=[k for k in ks if k[4] != "k2"], copies=nh2d, launches=ls)   # 2 kernels per launch: refused
    assert toy.check(str(g), str(bad)) == 1
    noeager = tmp_path / "noeager.sqlite"                                                  # eager work missing: refused
    census._mkdb(str(noeager), graphs=[(1000 * i, 1000 * i + 300, 50 + i, 7) for i in range(20)])
    assert toy.check(str(noeager), str(n)) == 1


def test_e4b_bracket_on_the_real_scheduler_with_staggered_admission():
    # round 2 B1: serve_paged admits one 512-token prefill chunk per step, so at B=16 the first row is 15 tokens ahead of
    # the last. With the registered 160-token rows the window fits; a 104-token budget refuses BEFORE the range opens.
    sched_mod = importlib.import_module("experts4bit_qlora.engines.scheduler")
    e4b = _mod("sc1b_e4b_census")

    class Runner:
        def run_prefill(self, chunks):
            return {rid: 1 for rid, s, t in chunks if s + t >= 512}

        def run_decode(self, rids):
            return {r: 1 for r in rids}

        def free_slot(self, rid):
            pass
    for b in (1, 16):
        for tokens, ok in ((160, True), (104, b == 1)):                 # 104 fits at B=1; staggered B=16 overruns it
            s = sched_mod.ContinuousScheduler(runner=Runner(), max_seqs=b, chunk_tokens=512)
            for _ in range(b):
                s.add_request([0] * 512, max_new_tokens=tokens)
            opened = []
            try:
                r = e4b.census_window(s, b, 32, 64, lambda: opened.append(1), lambda: None, lambda: None, tokens=tokens)
                assert ok and r["steps_bracketed"] == 64 and r["window_decode_positions"][1] <= tokens, r
            except e4b.Refusal as e:
                assert not ok and "retire it inside the window" in str(e) and not opened, (b, tokens, e)


def test_the_a1_evaluator_reads_the_reducers_own_gap_records(tmp_path):
    # A1 (#846): sc1b_read.py consumes exactly what census.gap() writes; a key rename on either side fails here
    rd = _mod("sc1b_read")
    rd.self_test()

    def arm(engine, batch, p, terms, labels=(), unprofiled=None, k_in=600):
        full = {c: 0.0 for c in census.CLASSES}
        full.update(I_in=0.0, idle_out=0.0)
        full.update(terms)
        full["O"] = round(sum(v for k, v in full.items() if k != "O") - p, 6)
        noise = {k: 0.001 for k in full}
        noise["O"] = None
        return {"engine": engine, "batch": batch, "status": "labelled" if labels else "ok", "labels": list(labels), "P_ms": p,
                "terms": full, "noise_iqr": noise, "unprofiled_ms": unprofiled or p, "node": {"status": "ok", "kernels_in_graph": k_in}}
    e1 = arm("e4b", 1, 4.5, {"moe_expert": 1.4, "attn": 0.5, "dense_gemm": 0.6, "I_in": 1.6, "idle_out": 0.4}, k_in=1100)
    l1 = arm("llamacpp", 1, 3.0, {"moe_expert": 1.3, "attn": 0.4, "dense_gemm": 0.6, "I_in": 0.4, "idle_out": 0.3}, k_in=650)
    e16 = arm("e4b", 16, 9.6, {"moe_expert": 5.0, "attn": 1.0, "I_in": 2.0, "idle_out": 1.6})
    l16 = arm("llamacpp", 16, 15.4, {"moe_expert": 5.5, "attn": 1.1, "I_in": 1.8, "idle_out": 7.0})
    v1 = arm("vllm", 1, 3.7, {"moe_expert": 1.2, "attn": 0.5, "dense_gemm": 0.6, "I_in": 0.9, "idle_out": 0.5})
    for name, rec in (("sc1b_arm_e4b_b1", e1), ("sc1b_arm_llamacpp_b1", l1), ("sc1b_arm_e4b_b16", e16), ("sc1b_arm_llamacpp_b16", l16),
                      ("sc1b_arm_vllm_b1", v1), ("sc1b_gap_e4b_llamacpp_b1", census.gap(e1, l1, 1.484)),
                      ("sc1b_gap_e4b_llamacpp_b16", census.gap(e16, l16, 0.624)), ("sc1b_gap_e4b_vllm_b1", census.gap(e1, v1, 1.203))):
        (tmp_path / f"{name}.json").write_text(json.dumps(rec))
    arms, gaps, v = rd.read(str(tmp_path))
    assert {q: r["verdict"] for q, r in v.items()} == {"Q1": "HOLDS", "Q2": "HOLDS", "Q3": "HOLDS", "Q4": "HOLDS", "Q5": "HOLDS"}, v
    assert v["Q2"]["ratio"] == round(1100 / 650, 4) and v["Q1"]["largest"] == "I_in" and v["Q4"]["share"] > 0.5
    md = rd.render(arms, gaps, v)
    assert "| G1 | llamacpp | 1 | ok |" in md and "| sglang | 16 | missing |" in md
    # a blocking label on either arm leaves the gap, and every prediction that reads it, UNREAD
    l1b = dict(l1, labels=["CLASS_MAP_INCOMPLETE"], status="labelled")
    (tmp_path / "sc1b_gap_e4b_llamacpp_b1.json").write_text(json.dumps(census.gap(e1, l1b, 1.484)))
    _arms, _gaps, v = rd.read(str(tmp_path))
    assert v["Q1"]["verdict"] == v["Q3"]["verdict"] == "UNREAD" and v["Q2"]["verdict"] == "HOLDS", v
