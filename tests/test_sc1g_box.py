"""Lane SC1g's box I (#846): the quality of each engine's gpt-oss-20b arithmetic on identical tokens. Executed, not grepped,
wherever the property is behaviour: the route record P42's hook writes (and does not write on a kill), the pinned chat date
in the K8 window, SGLang's gpt-oss quality engagement, the child environment, and the reducer's rule."""
import json
import os
import pathlib
import signal
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RUN = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
BOX = (REPO / "bench" / "sc2" / "sc1g_box_i.sh").read_text()
SGL = REPO / "bench" / "sc1" / "sglang" / "server.sh"


def test_box_i_is_wired_on_box_g_s_image_and_kernel_package():
    assert 'case "$BOX" in A|B|C|D|E|F|G|H|I|J|K|L|M) ;;' in RUN and "I) box_i;; J) box_j;; K) box_k;; L) box_l;; M) case "${SC1_SC5_PHASE:-read}" in ref) box_m_ref;; *) box_m;; esac;; esac" in RUN and '[ "$BOX" = I ] && prove_i' in RUN
    assert "G|I|J) GNF4_SHA=dc8f94abfd868f149178623f6eb403dc8b892b02;; esac" in RUN
    assert "I) . $W/sc2_box_e.sh; . $W/sc2g_box_g.sh; . $W/sc1g_box_i.sh; install_vllm; install_llamacpp ;;" in RUN
    assert 'I) PROVE_NEEDS="vllm llamacpp";;' in RUN   # A5: SGLang has no KL arm, so the proof does not require it
    assert "kl_full=[e4b_serve vllm llamacpp_q8] (A5; SGLang UNREAD by registration)" in RUN   # the rows prove_i's A5 proof reads
    assert "install_vllm; install_llamacpp ;;   # SC1g A5" in RUN and "quality=[e4b_serve e4b_nf4 vllm sglang_native" not in RUN
    # the route record needs e4b#1129's counters: the tripwire refuses an older e4b on box I
    assert 'if os.environ["TRIP_BOX"] in ("I", "J"):' in RUN and 'getattr(hr, "ROUTE_SEEN", None)' in RUN


def _child_env(box):
    start = RUN.index("# every lever is unset; each arm names its own switches")
    end = RUN.index(": > summary.txt")
    script = f'BOX={box}\nexport E4B_INT4_PREFILL=batched E4B_PAGED_PREFILL_ATTN=bogus\n{RUN[start:end]}\nenv\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)


def test_box_i_children_inherit_neither_prefill_route_pin():
    env = _child_env("I")
    assert "E4B_INT4_PREFILL" not in env and "E4B_PAGED_PREFILL_ATTN" not in env


def test_box_i_sources_under_set_u_with_only_w_defined():
    sc2 = REPO / "bench" / "sc2"
    files = " ".join(f'. "{sc2 / f}";' for f in ("sc2_box_e.sh", "sc2g_box_g.sh", "sc1g_box_i.sh"))
    out = subprocess.run(["bash", "-c", f"set -uo pipefail; W=$(mktemp -d); {files} echo SOURCED"], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0 and "SOURCED" in out.stdout, out.stderr


def test_box_i_pins_each_engine_s_quality_path():
    # e4b: serve_paged's gpt-oss env through the hook, the NF4 control, SC1's K8 args, the window from its file, the floor pair
    assert 'SC1G_E4B_SERVE="E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=1 E4B_SERVE_ATTN_INT4_CALIB=0' in BOX
    assert 'SC1G_E4B_NF4="E4B_SERVE_EXP_INT4=0' in BOX
    assert "--ppl-oracle eager --ppl-chunk 64" in BOX and "--ppl-oracle eager --ppl-chunk 128" in BOX
    assert "SC1G_CHAT_DATE=$SC1G_CHAT_DATE" in BOX and "SC1G_CHAT_DATE=2026-10-05" in BOX
    assert "SC1G_WINDOW_FILE=$W/sc1g/k8_window_$SRC.json" in BOX and "SC1G_ROUTE_OUT=$W/sc1g/routes/$NAME" in BOX
    assert "$W/sc1g_k8.py k8 -- --model" in BOX and "$W/sc1g_k8.py windows --model" in BOX
    # A1: the graded texts, the control, the pinned dataset, the diagnostics, the kernel check with its mutation
    assert 'SC1G_SRCS="conv1 conv2"' in BOX and 'SC1G_CTRL="wikitext"' in BOX
    assert "SC1G_UC_REPO=HuggingFaceH4/ultrachat_200k; SC1G_UC_REV=8049631c405ae6576f93f445c6b8166f76f5505a" in BOX
    assert '"$SC1G_E4B_SERVE GNF4_PDL=0"' in BOX and '"$SC1G_E4B_SERVE $SC1G_NOFOLD"' in BOX and "--ppl-chunk 1" in BOX
    assert 'SC1G_NOFOLD="E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0"' in BOX
    assert "env $ROUTEENV $FOLDS $STACK" in BOX                       # an arm's stack overrides SC1's folds
    assert "SC1G_CAPTURE_OUT=$W/sc1g/capture_$SRC.pt" in BOX and "--synthetic --layers 0 --mutate" in BOX
    # vLLM on gpt-oss: Marlin W4A16, TRITON_ATTN (FA2 has no sinks); SGLang's two modes; llama.cpp's q8 control
    assert "SC1_MODEL=$SC2G_MID SC1_REV=$SC2G_REV SC1_ATTN_BACKEND=TRITON_ATTN SC1_MOE_BACKEND=marlin" in BOX
    assert 'echo gptoss_qm || echo gptoss_q' in BOX
    assert '$([ "$V" = q8 ] && echo GGML_CUDA_MMQ_PREC=q8)' in BOX and 'r["mmq_prec_env"]' in BOX


K8 = REPO / "bench" / "sc2" / "sc1g_k8.py"


def _wrapped(tmp_path, name, body, env_extra=None):
    """Run sc1g_k8.py k8 against a stand-in step_decomp.py (beside a copy of the wrapper) whose main() is `body`."""
    tmp_path = tmp_path / name
    tmp_path.mkdir()
    (tmp_path / "sc1g_k8.py").write_text(K8.read_text())
    (tmp_path / "step_decomp.py").write_text("def _k8_window(a, tok):\n    raise SystemExit('the original window builder ran')\n"
                                             "def main():\n" + "".join(f"    {ln}\n" for ln in body.splitlines()))
    out_dir = tmp_path / "routes"
    out_dir.mkdir()
    env = dict(os.environ, SC1G_ROUTE_OUT=str(out_dir / "arm"), PYTHONPATH=str(REPO), **(env_extra or {}))
    out = subprocess.run([sys.executable, str(tmp_path / "sc1g_k8.py"), "k8", "--", "--x"], capture_output=True, text=True,
                         env=env, timeout=120)
    return out.returncode, sorted(p.name for p in out_dir.iterdir()), out_dir, out.stdout + out.stderr


def test_the_wrapper_records_routes_only_from_an_e4b_process_and_never_on_a_kill(tmp_path):
    rc, files, _, log = _wrapped(tmp_path, "a", "import sys; assert sys.argv[1:] == ['--x']")
    assert rc == 0 and files == [], log                                # step_decomp never loaded e4b: nothing written
    rc, files, d, log = _wrapped(tmp_path, "b", "from experts4bit_qlora.engines import hot_residency as hr\n"
                                                "hr._seen_route('mxfp4_gemv', 4); hr._seen_route('nf4_mtile_host', 2048)")
    assert rc == 0 and len(files) == 1 and files[0].startswith("arm.") and files[0].endswith(".json"), log
    assert json.loads((d / files[0]).read_text())["route_seen"] == {"mxfp4_gemv|le256": 1, "nf4_mtile_host|gt256": 1}
    rc, files, _, _ = _wrapped(tmp_path, "c", "from experts4bit_qlora.engines import hot_residency as hr\nimport os, signal\n"
                                              "hr._seen_route('mxfp4_gemv', 4); os.kill(os.getpid(), signal.SIGALRM)")
    assert rc == -signal.SIGALRM and files == []                       # perl alarm's death writes no record: the reducer FAILs it
    rc, files, _, _ = _wrapped(tmp_path, "e", "from experts4bit_qlora.engines import hot_residency as hr\n"
                                              "hr._seen_route('mxfp4_gemv', 4); raise SystemExit(3)")
    assert rc == 3 and len(files) == 1                                 # an ordinary error exit still records what ran


def _window(path, ids, prompt_len=4, steps=3):
    import hashlib
    import struct
    sha = hashlib.sha256(struct.pack(f"<{prompt_len + steps + 1}q", *ids[:prompt_len + steps + 1])).hexdigest()
    path.write_text(json.dumps({"ids": ids, "text_sha": sha, "prompt_len": prompt_len, "steps": steps, "source": "conv1"}))
    return sha


def test_the_e4b_arm_scores_the_window_file_and_refuses_one_whose_ids_do_not_hash_to_its_sha(tmp_path):
    pytest.importorskip("torch")
    win = tmp_path / "w.json"
    sha = _window(win, [5, 6, 7, 8, 9, 10, 11, 12])
    body = ("import types\na = types.SimpleNamespace(prompt_len=4, ppl_steps=3, batch=1)\n"
            "ids, step, prompts, ppl, s = _k8_window(a, None)\nprint('WINDOW_SHA', s, ids.tolist(), prompts)")
    rc, _, _, log = _wrapped(tmp_path, "w", body, {"SC1G_WINDOW_FILE": str(win)})
    assert rc == 0 and f"WINDOW_SHA {sha} [5, 6, 7, 8, 9, 10, 11, 12] [[5, 6, 7, 8]]" in log, log
    rec = json.loads(win.read_text())
    rec["ids"][2] = 99                                                 # the ids no longer hash to the recorded sha
    win.write_text(json.dumps(rec))
    rc, _, _, log = _wrapped(tmp_path, "w2", body, {"SC1G_WINDOW_FILE": str(win)})
    assert rc != 0 and "hash to" in log, log
    rc, _, _, log = _wrapped(tmp_path, "w3", body)                     # unset: step_decomp's own builder runs
    assert rc != 0 and "the original window builder ran" in log, log


class _RoleTok:
    """ids: 1 <|start|>, 2 <|message|>, 3 <|end|>, 4 <|return|>, 5 <|channel|>; 10 'user', 11 'assistant', 12 'final', >= 100 text."""
    names = {"<|start|>": 1, "<|message|>": 2, "<|end|>": 3, "<|return|>": 4, "<|channel|>": 5}

    def convert_tokens_to_ids(self, s):
        return self.names[s]

    def decode(self, ids):
        return "".join({10: "user", 11: "assistant", 12: "final", 5: "<|channel|>"}.get(i, "?") for i in ids)


def test_the_window_roles_come_from_the_template_s_special_tokens():
    sys.path.insert(0, str(K8.parent))
    try:
        import sc1g_k8
        ids = [1, 10, 2, 100, 101, 3, 1, 11, 5, 12, 2, 102, 103, 104, 4]
        r = sc1g_k8.roles(_RoleTok(), ids)
        assert r == ["markup"] * 3 + ["user"] * 2 + ["markup"] * 6 + ["assistant"] * 3 + ["markup"], r
    finally:
        sys.path.remove(str(K8.parent))


def test_the_gemv_check_aggregates_and_reads_its_mutation():
    pytest.importorskip("torch")
    sys.path.insert(0, str(K8.parent))
    try:
        import torch
        import sc1g_gemv_check as g
        assert g.rel(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])) == 0.0
        rows = [{"layer": 0, "which": "gu", "kernel_vs_exact_bf16": 1e-5, "kernel_vs_exact": 1e-3, "scheme_exact_vs_raw": 0.01,
                 "block_crest_max": 5.0},
                {"layer": 0, "which": "gu", "kernel_vs_exact_bf16": 2e-5, "kernel_vs_exact": 2e-3, "scheme_exact_vs_raw": 0.03,
                 "block_crest_max": 30.0}]
        s = g.summarize(rows)["L0_gu"]
        assert s["n"] == 2 and s["kernel_vs_exact_bf16_max"] == 2e-5 and abs(s["scheme_mean"] - 0.02) < 1e-12 and s["crest_max"] == 30.0
    finally:
        sys.path.remove(str(K8.parent))


def test_the_chat_date_pin_reaches_transformers_strftime_now_and_refuses_a_bad_day():
    pytest.importorskip("transformers")
    sys.path.insert(0, str(K8.parent))
    try:
        import sc1g_k8
        import transformers.utils.chat_template_utils as ctu
        orig = ctu.datetime
        try:
            assert sc1g_k8.pin_chat_date("2031-01-02") == "2031-01-02"
            env = ctu._compile_jinja_template.__wrapped__("{{ strftime_now('%Y-%m-%d') }}") \
                if hasattr(ctu._compile_jinja_template, "__wrapped__") else ctu._compile_jinja_template("{{ strftime_now('%Y-%m-%d') }}")
            assert env.render() == "2031-01-02"
            with pytest.raises(SystemExit):                              # a second pin would stack: refused
                sc1g_k8.pin_chat_date("2031-01-03")
        finally:
            ctu.datetime = orig
        with pytest.raises(SystemExit):
            sc1g_k8.pin_chat_date("Oct 5")
        if not os.environ.get("SC1G_CHAT_DATE"):
            assert sc1g_k8.pin_chat_date("") is None                   # unset: nothing pinned
    finally:
        sys.path.remove(str(K8.parent))


def test_the_shared_harness_files_are_untouched():
    """SC1g must not change P39's step_decomp, SC1's sc1_prompts or P42's hook: other lanes pin their bytes."""
    out = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "origin/main", "--", "bench/p39/step_decomp.py",
                          "bench/sc1/sc1_prompts.py", "bench/p42/hook/usercustomize.py"], capture_output=True, timeout=60)
    if out.returncode == 128:
        pytest.skip("no origin/main to compare against")
    assert out.returncode == 0, "a shared, cross-pinned harness file differs from main"


def _sglang_engagement(mode, info, log_text, tmp_path):
    src = SGL.read_text()
    body = src[src.index("<<'PYT'\n") + len("<<'PYT'\n"): src.index("\nPYT\n", src.index("<<'PYT'\n"))]
    log = tmp_path / "srv.log"
    log.write_text(log_text)
    (tmp_path / "srv.log.server_info.json").write_text(json.dumps(info))
    env = {"SC1_MODE": mode, "SC1_MFS": "0.75", "SC1_LOG": str(log), "SC1_STARTUP_S": "30", "SC1_JIT": str(tmp_path / "jit"),
           "SC1_CMD": "x", "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, "-c", body], capture_output=True, text=True, env=env, timeout=60)
    return out.returncode, out.stdout


def test_sglang_gptoss_quality_modes_demand_radix_on_one_request_and_their_moe_runner(tmp_path):
    info = {"version": "0.5.20", "attention_backend": "triton", "disable_radix_cache": False, "max_running_requests": 1,
            "moe_runner_backend": "flashinfer_mxfp4", "mem_fraction_static": 0.75, "max_total_num_tokens": 9000,
            "context_length": 4096, "dtype": "bfloat16"}
    rc, out = _sglang_engagement("gptoss_q", info, "The server is fired up and ready to roll!\n", tmp_path)
    assert rc == 0 and '"mode": "gptoss_q"' in out, out
    rc, out = _sglang_engagement("gptoss_qm", info, "", tmp_path)                    # marlin requested, native resolved
    assert rc == 43 and "not marlin" in out, out
    rc, out = _sglang_engagement("gptoss_qm", dict(info, moe_runner_backend="marlin"), "", tmp_path)
    assert rc == 0, out
    rc, out = _sglang_engagement("gptoss_q", dict(info, disable_radix_cache=True), "", tmp_path)   # served shape needs the cache
    assert rc == 43 and "disable_radix_cache" in out, out
    flags = subprocess.run(["bash", "-c", f'. "{SGL}"; SC1_SGLANG_MEM_FRACTION_STATIC=0.75 sglang_server_flags gptoss_qm 30000'],
                           capture_output=True, text=True, timeout=30).stdout.split()
    assert flags.count("--moe-runner-backend") == 2 and flags[-2:] == ["--moe-runner-backend", "marlin"], flags   # argparse: the last wins


def test_the_sc1g_reducer_self_test_passes():
    out = subprocess.run([sys.executable, str(REPO / "bench" / "sc2" / "sc1g_reduce.py"), "--self-test"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "self-test OK (55 cases)" in out.stdout, out.stdout + out.stderr


def test_the_capture_keeps_the_selected_layers_gate_up_and_down_per_decode_step(tmp_path):
    """Stand-in gnf4 modules: two decode steps of 24 layers x (gate_up, down) GEMV calls, each after a quant_x_rows."""
    pytest.importorskip("torch")
    (tmp_path / "int4_b32.py").write_text("def quant_x_rows(x):\n    return x, None\n")
    (tmp_path / "mxfp4_grouped.py").write_text("def gemv_mxfp4_b32(xq, xs, blocks, scales, eids, N, K, part=None):\n    return xq\n")
    out = tmp_path / "cap.pt"
    script = f"""
import sys, torch
sys.path[:0] = [{str(tmp_path)!r}, {str(K8.parent)!r}]
import sc1g_k8
sc1g_k8.capture({str(out)!r}, steps=1, layers=(0, 23))
import int4_b32, mxfp4_grouped
for step in range(2):
    for layer in range(24):
        for which, N in (("gu", 5760), ("dn", 2880)):
            x = torch.full((4, 8), float(step * 100 + layer))
            xq, xs = int4_b32.quant_x_rows(x)
            mxfp4_grouped.gemv_mxfp4_b32(xq, xs, None, None, torch.arange(4), N, 2880)
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    import torch
    cap = torch.load(out, weights_only=False)
    got = [(c["step"], c["layer"], c["which"], c["N"], float(c["x"][0, 0])) for c in cap["rows"]]
    assert got == [(0, 0, "gu", 5760, 0.0), (0, 0, "dn", 2880, 0.0), (0, 23, "gu", 5760, 23.0), (0, 23, "dn", 2880, 23.0)], got
    assert cap["gemv_calls"] == 96


def test_box_j_is_the_e4b_only_diagnostic_box_with_its_decisive_arms_first():
    """A2/A3: box J installs no comparator, needs no proof (guard <= 1 h), runs A3's split in priority order -- conv1's K1/K2/K5
    rows, e4b#1175's attention check, the K2/K5 triple on conv2-conv4, then the controls -- and builds four conversations."""
    assert "  J) . $W/sc2_box_e.sh; . $W/sc2g_box_g.sh; . $W/sc1g_box_i.sh ;;" in RUN
    assert 'J) PROVE_NEEDS="";;' in RUN and "J) box_j;; K) box_k;; L) box_l;; M) case "${SC1_SC5_PHASE:-read}" in ref) box_m_ref;; *) box_m;; esac;; esac" in RUN and "G|I|J) GNF4_SHA=" in RUN
    body = BOX[BOX.index("box_j_a3(){"):BOX.index("i_arms_a6(){")]   # A3's box J, kept for the record (box J now runs A6)
    assert "install_" not in body and "fetch_gptoss_gguf" not in body
    order = [body.index(s) for s in ("e4b_serve_v1_conv1 ", "e4b_mxpre_prefill128_conv1 ", "e4b_nf4_prefill128_conv1 ",
                                     "e4b_serve_served_conv1 ", "e4b_nf4_served_conv1 ", "sc1g_attn_check.py",
                                     "for c in conv2 conv3 conv4", "e4b_serve_kvg4_conv1 ", "e4b_serve_nofold_conv1 ",
                                     "e4b_serve_pdl0_conv1 ", "e4b_mxpre_prefill128_conv2 ", "e4b_nf4_prefill128_conv2 ")]
    assert order == sorted(order), order
    loop = body[body.index("for c in conv2 conv3 conv4"):body.index("e4b_serve_kvg4_conv1 ")]
    for stem, env in (("e4b_serve_served_$c", '"$SC1G_E4B_SERVE"'), ("e4b_nf4_served_$c", '"$SC1G_E4B_NF4"'),
                      ("e4b_serve_v1_$c", '"$SC1G_E4B_SERVE E4B_MXFP4_GEMV=0"')):
        assert f"{stem} {env} $c" in loop
    assert "SC1G_NCONV=4 i_windows" in body and "--n-conv ${SC1G_NCONV:-2}" in BOX
    assert "can_run 600 attn_check" in body and "SC1G_ATTN_CHECK ERROR" in body
    assert 'SC1G_E4B_MXPRE="E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=0' in BOX and "--kv-groups 16" not in body


def test_box_j_children_inherit_neither_prefill_route_pin():
    env = _child_env("J")
    assert "E4B_INT4_PREFILL" not in env and "E4B_PAGED_PREFILL_ATTN" not in env


# ---- A4: the fidelity instrument's engine side ------------------------------------------------------------------------

def test_box_i_under_a5_runs_the_full_kl_arms_first_on_four_conversations():
    body = BOX[BOX.index("box_i(){"):BOX.index("# ---- the proof (SC1_PROVE=1), A5")]
    assert "SC1G_NCONV=4 i_windows || finish 19; i_ref_full_stage" in body and "i_arms_a5" in body
    arms = BOX[BOX.index("i_arms_a5(){"):]
    order = [arms.index(s) for s in ("i_e4b_full e4b_serve_served_$SRC", "i_e4b_full e4b_nf4_served_$SRC", "i_vllm_full $SRC",
                                     "i_ll_full $V $SRC", "A5DESC")]
    assert order == sorted(order), order
    assert "i_sgl" not in arms[:arms.index("A5DESC")]          # SGLang is UNREAD under A5: no KL arm


def test_every_full_kl_arm_refuses_without_a_registered_reference():
    for fn in ("i_e4b_full", "i_vllm_full", "i_ll_full"):
        assert "SHA=$(i_ref_full $SRC) || { i_nofull" in BOX[BOX.index(f"{fn}(){{"):BOX.index(f"{fn}(){{") + 500], fn
    ref = BOX[BOX.index("i_ref_full(){"):BOX.index("i_ref_full_stage(){")]
    assert "ref_full_shas.json" in ref and 'sha256sum "$F"' in ref and '[ "$GOT" = "$WANT" ]' in ref
    assert "SC1G_REF_FULL_FILE=$SC1G_REF_FULL_DIR/ref_full_$SRC.npy SC1G_REF_FULL_SHA=$SHA SC1G_KL_OUT=" in BOX
    assert "SC1_REF_FULL=$SC1G_REF_FULL_DIR/ref_full_$SRC.npy SC1_REF_FULL_SHA=$SHA SC1_KL_OUT=" in BOX
    assert "--ref-full $W/sc1g/ref_full_$SRC.f16 --kl-out $W/sc1g/kl_$S.bin" in BOX


def test_the_a5_proof_reads_one_full_kl_row_per_engine_path():
    pv = BOX[BOX.index("prove_i(){"):BOX.index("# ---- box J (SC1_BOX=J)")]
    for s in ("i_e4b_full e4b_serve_served_conv1", "i_vllm_full conv1", "i_ll_full q8 conv1", "--prove-a5",
              "sc1g_kl.py --self-test", "i_ref_full conv1 > /dev/null || {"):
        assert s in pv, s


def test_every_named_arm_refuses_without_a_registered_reference():
    for fn in ("i_e4b_named", "i_vllm_named", "i_sgl_named", "i_ll_named"):
        body = BOX[BOX.index(f"{fn}(){{"):]
        body = body[:body.index("\n}") if "\n}" in body[:400] else 400]
        assert "SHA=$(i_ref $SRC) || { i_noref" in BOX[BOX.index(f"{fn}(){{"):BOX.index(f"{fn}(){{") + 400], fn
    ref = BOX[BOX.index("i_ref(){"):BOX.index("i_ref_stage(){")]
    assert "ref_shas.json" in ref and 'sha256sum "$F"' in ref and '[ "$GOT" = "$WANT" ]' in ref
    assert "SC1G_REF_FILE=$SC1G_REF_DIR/ref_$SRC.npz SC1G_REF_SHA=$SHA SC1G_NAMED_OUT=$W/sc1g/named_$NAME.npz" in BOX
    assert "SC1_NAMED_REF=$SC1G_REF_DIR/ref_$SRC.npz SC1_NAMED_REF_SHA=$SHA" in BOX
    assert "--named-ref $SC1G_REF_DIR/ref_$SRC.npz --named-ref-sha $SHA" in BOX
    assert "--named $W/sc1g/named_ids_$SRC.bin --named-out $W/sc1g/named_$S.bin --named-k 64" in BOX


def test_the_named_scorer_hooks():
    sys.path.insert(0, str(REPO / "bench" / "sc1" / "vllm"))
    src = (REPO / "bench" / "sc1" / "vllm" / "sc1_vllm_common.py").read_text()
    ns = {}
    exec(compile(src[src.index("def served_sampling_kwargs"):src.index("def served_row")], "x", "exec"), ns)
    kw = ns["served_sampling_kwargs"]("named", 7, [3, 7, 9])
    assert kw["logprob_token_ids"] == [3, 7, 9] and ns["served_sampling_kwargs"]("named", 5, [3, 9])["logprob_token_ids"] == [3, 9, 5]
    sg = (REPO / "bench" / "sc1" / "sglang" / "sc1_sglang_nll.py").read_text()
    ns2 = {}
    exec(compile(sg[sg.index("def served_request_ids"):sg.index("def score_served(")], "y", "exec"), ns2)
    assert ns2["served_request_ids"](7, [3, 7, 9]) == [7, 3, 9] and ns2["served_request_ids"](7) == [7]
    cpp = (REPO / "bench" / "sc1" / "llamacpp" / "nll_teacher_forced.cpp").read_text()
    assert cpp.count("named_lps(lg, n_vocab,") == 2 and '"--named-out"' in cpp and "out of vocab" in cpp
    vn = (REPO / "bench" / "sc1" / "vllm" / "sc1_vllm_nll.py").read_text()
    assert 'if C.env("REF_FULL") and mode == "served":\n        lp_req = "full"' in vn and "A5 never downgrades" in vn
    assert cpp.count("kl_row(&kl_buf[") == 2 and "constexpr size_t KLW = 5;" in cpp and '"--ref-full"' in cpp and "is not exactly %d x %d fp16" in cpp
    # box I's converter reads the harness's five columns into the same support arrays the e4b and vLLM records carry
    bi = (REPO / "bench" / "sc2" / "sc1g_box_i.sh").read_text()
    assert "reshape(-1, 5)" in bi and "reshape(-1, 2)" not in bi
    for k in ("eng_kl=a[:, 0]", "eng_target_lp=a[:, 1]", "eng_kl_common=a[:, 2]", "eng_masked_mass=a[:, 3]", "eng_n_masked=a[:, 4]"):
        assert k in bi, k
    for src in (vn, (REPO / "bench" / "sc2" / "sc1g_k8.py").read_text()):
        assert "kl_full_support(" in src and "kl_full_rows(" not in src and "eng_masked_mass=" in src
    drv = (REPO / "bench" / "sc1" / "sc1_drive.sh").read_text()
    assert 'if [ -n "${SC1G_REF_FULL_SRC:-}" ]; then' in drv and '$SSH "mkdir -p $W/sc1g_ref_full"' in drv


@pytest.mark.parametrize("free,staged,want_rc,note", [
    (317, 4, 0, "disk: 317 GB free + 4 GB the lane staged"),   # sc1g-prove-a5-1/-2's healthy host, now admitted
    (317, None, 13, "refused: disk 317 GB (+0 staged)"),         # the same free space with nothing staged is still refused
    (320, None, 0, None),                                        # the floor itself, unchanged
    (300, 4, 13, "refused: disk 300 GB (+4 staged)"),            # staged bytes never rescue a genuinely small disk
])
def test_the_disk_floor_counts_only_the_lane_s_own_staged_inputs(tmp_path, free, staged, want_rc, note):
    """SC1g A5: the launcher orders a fixed 320 GB disk, so the box's floor counts the reference rows the controller staged
    (sc1g_ref_full/) toward MIN_DISK_GB instead of refusing a healthy host on the lane's account. The block is run as
    sc1_run.sh has it, with df/du faked on PATH."""
    import subprocess
    blk = RUN[RUN.index("FREE_GB=$(df -BG --output=avail $W"):]
    blk = blk[:blk.index("finish 13; }") + len("finish 13; }")]
    assert "MIN_DISK_GB" in blk and "STAGED_GB" in blk and blk.count("finish 13") == 1
    w, fake = tmp_path / "w", tmp_path / "bin"
    w.mkdir()
    fake.mkdir()
    (fake / "df").write_text(f"#!/bin/sh\necho ' Avail'\necho ' {free}G'\n")
    (fake / "du").write_text(f"#!/bin/sh\nprintf '{staged or 0}G\\t%s\\n' \"$4\"\n")
    for f in ("df", "du"):
        (fake / f).chmod(0o755)
    if staged is not None:
        (w / "sc1g_ref_full").mkdir()
    script = (f"set -uo pipefail\nW={w}\nMIN_DISK_GB=320\ncd {w}\n"
              "say(){ echo \"say: $*\"; }\nfinish(){ echo \"finish $1\"; exit $1; }\n" + blk + "\necho passed\n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={"PATH": f"{fake}:/usr/bin:/bin"})
    assert r.returncode == want_rc, r.stdout + r.stderr
    got = ((w / "summary.txt").read_text() if (w / "summary.txt").exists() else "") + ((w / "REFUSAL").read_text() if (w / "REFUSAL").exists() else "")
    if note:
        assert note in got, got
    if want_rc == 0:
        assert "passed" in r.stdout and not (w / "REFUSAL").exists()

