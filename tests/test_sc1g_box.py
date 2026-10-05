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
    assert 'case "$BOX" in A|B|C|D|E|F|G|H|I) ;;' in RUN and "I) box_i;; esac" in RUN and '[ "$BOX" = I ] && prove_i' in RUN
    assert "G|I) GNF4_SHA=dc8f94abfd868f149178623f6eb403dc8b892b02;; esac" in RUN
    assert "I) . $W/sc2_box_e.sh; . $W/sc2g_box_g.sh; . $W/sc1g_box_i.sh; install_vllm; install_sglang; install_llamacpp ;;" in RUN
    assert 'I) PROVE_NEEDS="vllm sglang llamacpp";;' in RUN
    assert "quality=[e4b_serve e4b_nf4 vllm sglang_native sglang_marlin llamacpp llamacpp_q8]" in RUN
    # the route record needs e4b#1129's counters: the tripwire refuses an older e4b on box I
    assert 'if os.environ["TRIP_BOX"] == "I":' in RUN and 'getattr(hr, "ROUTE_SEEN", None)' in RUN


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
    # e4b: serve_paged's gpt-oss env through the hook, the NF4 control, SC1's K8 args, the chat window, the floor pair
    assert 'SC1G_E4B_SERVE="E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=1 E4B_SERVE_ATTN_INT4_CALIB=0' in BOX
    assert 'SC1G_E4B_NF4="E4B_SERVE_EXP_INT4=0' in BOX
    assert "--ppl-chat --ppl-chat-suffix \"$SC1G_CHAT_SUFFIX\"" in BOX and "SC1G_CHAT_SUFFIX='<|channel|>final<|message|>'" in BOX
    assert "--ppl-oracle eager --ppl-chunk 64" in BOX and "--ppl-oracle eager --ppl-chunk 128" in BOX
    assert "SC1G_CHAT_DATE=$SC1G_CHAT_DATE" in BOX and "SC1G_CHAT_DATE=2026-10-05" in BOX
    assert "SC1G_ROUTE_OUT=$W/sc1g/routes/$NAME" in BOX and "$W/sc1g_k8.py k8 -- --model" in BOX
    assert "$W/sc1g_k8.py windows --model" in BOX
    # vLLM on gpt-oss: Marlin W4A16, TRITON_ATTN (FA2 has no sinks); SGLang's two modes; llama.cpp's q8 control
    assert "SC1_MODEL=$SC2G_MID SC1_REV=$SC2G_REV SC1_ATTN_BACKEND=TRITON_ATTN SC1_MOE_BACKEND=marlin" in BOX
    assert 'echo gptoss_qm || echo gptoss_q' in BOX
    assert '$([ "$V" = q8 ] && echo GGML_CUDA_MMQ_PREC=q8)' in BOX and 'r["mmq_prec_env"]' in BOX


K8 = REPO / "bench" / "sc2" / "sc1g_k8.py"


def _wrapped(tmp_path, name, body):
    """Run `body` as the script sc1g_k8.py k8 would hand to runpy: a stand-in step_decomp.py beside a copy of the wrapper."""
    tmp_path = tmp_path / name
    tmp_path.mkdir()
    (tmp_path / "sc1g_k8.py").write_text(K8.read_text())
    (tmp_path / "step_decomp.py").write_text(body)
    out_dir = tmp_path / "routes"
    out_dir.mkdir()
    env = dict(os.environ, SC1G_ROUTE_OUT=str(out_dir / "arm"), PYTHONPATH=str(REPO))
    out = subprocess.run([sys.executable, str(tmp_path / "sc1g_k8.py"), "k8", "--", "--x"], capture_output=True, text=True,
                         env=env, timeout=120)
    return out.returncode, sorted(p.name for p in out_dir.iterdir()), out_dir


def test_the_wrapper_records_routes_only_from_an_e4b_process_and_never_on_a_kill(tmp_path):
    rc, files, _ = _wrapped(tmp_path, "a", "import sys; assert sys.argv[1:] == ['--x']")
    assert rc == 0 and files == []                                    # step_decomp never loaded e4b: nothing written
    rc, files, d = _wrapped(tmp_path, "b", "from experts4bit_qlora.engines import hot_residency as hr\n"
                                           "hr._seen_route('mxfp4_gemv', 4); hr._seen_route('nf4_mtile_host', 2048)")
    assert rc == 0 and len(files) == 1 and files[0].startswith("arm.") and files[0].endswith(".json")
    assert json.loads((d / files[0]).read_text())["route_seen"] == {"mxfp4_gemv|le256": 1, "nf4_mtile_host|gt256": 1}
    rc, files, _ = _wrapped(tmp_path, "c", "from experts4bit_qlora.engines import hot_residency as hr\nimport os, signal\n"
                                           "hr._seen_route('mxfp4_gemv', 4); os.kill(os.getpid(), signal.SIGALRM)")
    assert rc == -signal.SIGALRM and files == []                      # perl alarm's death writes no record: the reducer FAILs it
    rc, files, _ = _wrapped(tmp_path, "e", "from experts4bit_qlora.engines import hot_residency as hr\n"
                                           "hr._seen_route('mxfp4_gemv', 4); raise SystemExit(3)")
    assert rc == 3 and len(files) == 1                                 # an ordinary error exit still records what ran


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
    assert out.returncode == 0 and "self-test OK (9 cases)" in out.stdout, out.stdout + out.stderr
