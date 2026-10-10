# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC2d's box K (#846): the E4B_PAGED_BULK_KV default's engagement reads, checked rather than grepped where they
can be run.

* ``k_arch`` (a Python heredoc in ``bench/sc2/sc2d_box_k.sh``) reads a checkpoint's own facts. It is run here on a
  hybrid-shaped and a sinks-shaped snapshot, and its record must pass the reducer's ARCH gate for the right model only;
* the reducer's registered rule passes its own self-test;
* box K is wired into the shared harness at every point box H is, and its Qwen3.6 pin is P98's.
"""
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SC2 = REPO / "bench" / "sc2"
BOX = (SC2 / "sc2d_box_k.sh").read_text()
RUN = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
DRIVE = (REPO / "bench" / "sc1" / "sc1_drive.sh").read_text()


def _heredoc(fn: str, tag: str) -> str:
    start = BOX.index(fn + "(){")
    body = BOX[BOX.index(f"<<'{tag}'", start):]
    body = body[body.index("\n") + 1:]
    return body[:body.index(f"\n{tag}\n")]


def _reducer():
    spec = importlib.util.spec_from_file_location("sc2d_reduce", SC2 / "sc2d_reduce.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _arch(tmp_path, model, config, weight_names):
    snap = tmp_path / f"snap_{model}"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps(config))
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "a.safetensors" for n in weight_names}}))
    script = tmp_path / "arch.py"
    script.write_text(_heredoc("k_arch", "PYA"))
    out = tmp_path / f"arch_{model}.json"
    r = subprocess.run([sys.executable, str(script), model, str(snap), str(out)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("SC2D_ARCH ")
    return json.loads(out.read_text())


def test_k_arch_reads_a_hybrid_s_attention_subset(tmp_path):
    types = (["linear_attention"] * 3 + ["full_attention"]) * 10          # Qwen3.6's 40 layers, 1 in 4 attention
    cfg = {"model_type": "qwen3_5_moe", "text_config": {"num_hidden_layers": 40, "layer_types": types}}
    names = [f"model.language_model.layers.{i}.mlp.gate.weight" for i in range(40)]
    a = _arch(tmp_path, "qw36", cfg, names)
    assert a["n_layers"] == 40 and a["attention_layers"] == list(range(3, 40, 4)) and a["linear_layers"] == 30
    red = _reducer()
    assert red.arch_ok("qw36", a)["ok"]
    assert not red.arch_ok("gptoss", a)["ok"]                            # no sinks: not the gpt-oss read


def test_k_arch_reads_gpt_oss_sinks_on_every_layer(tmp_path):
    types = ["sliding_attention", "full_attention"] * 12
    cfg = {"model_type": "gpt_oss", "num_hidden_layers": 24, "layer_types": types}
    names = [f"model.layers.{i}.self_attn.sinks" for i in range(24)] + [f"model.layers.{i}.self_attn.q_proj.weight"
                                                                       for i in range(24)]
    a = _arch(tmp_path, "gptoss", cfg, names)
    assert a["sinks_layers"] == 24 and len(a["attention_layers"]) == 24 and a["linear_layers"] == 0
    red = _reducer()
    assert red.arch_ok("gptoss", a)["ok"]
    assert not red.arch_ok("qw36", a)["ok"]                              # all attention: not a hybrid
    # a checkpoint missing a layer's sinks is not what the read claims
    p = _arch(tmp_path, "gptoss_partial", cfg, names[1:])
    assert p["sinks_layers"] == 23 and not red.arch_ok("gptoss", p)["ok"]


def test_box_k_refuses_an_e4b_without_the_fallback_counter_before_anything_is_fetched():
    """The ENGAGED gate reads ``flush_bulk_fallback`` (#1174): an older e4b would read <missing> on every server and
    report a false NOT_ENGAGED, so box K refuses it (rc 9) before any fetch, in the lane and in the proof."""
    trip = BOX[BOX.index("k_tripwire(){"):BOX.index("# k_prompts")]
    assert "flush_bulk_fallback" in trip and "finish 9" in trip
    assert "box_k(){\n  k_tripwire\n  phase 0" in BOX and "prove_k(){\n  k_tripwire\n" in BOX


def test_the_reducer_passes_its_self_test():
    r = subprocess.run([sys.executable, str(SC2 / "sc2d_reduce.py"), "--self-test"], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0 and "self-test OK" in r.stdout, r.stdout + r.stderr


def test_box_k_is_wired_like_box_h():
    assert 'case "$BOX" in A|B|C|D|E|F|G|H|I|J|K|L|M) ;;' in RUN
    assert 'case "$BOX" in K) GNF4_SHA=dc8f94abfd868f149178623f6eb403dc8b892b02;; esac' in RUN    # v0.41.0, as H
    assert '[ "$BOX" = J ] || [ "$BOX" = K ] || [ "$BOX" = L ]; then   # SC2g (box G), SC2c (box H), SC1g (boxes I, J), SC2d' in RUN
    assert "C|D|E|F|G|H|I|J|K|L|M) BASEPY=python3;;" in RUN
    torch = next(x for x in RUN.splitlines() if '"torch==2.8.0"' in x and "pipx logs/pip_torch.log" in x)
    assert '[ "$BOX" = K ]' in torch                                     # A1's lesson: every python3 box pins torch
    assert RUN.count('os.environ["TRIP_BOX"] in ("F", "G", "H", "I", "J", "K", "L")') == 2
    assert "  K) . $W/sc2_box_e.sh; . $W/sc2g_box_g.sh; . $W/sc2d_box_k.sh; install_sc2_client ;;" in RUN
    assert 'K) PROVE_NEEDS="sc2client";;' in RUN and '[ "$BOX" = K ] && prove_k' in RUN and "K) box_k;; L) box_l;; M) case "${SC1_SC5_PHASE:-read}" in ref) box_m_ref;; *) box_m;; esac;; esac" in RUN
    assert 'case "$SC1_BOX" in A|B|C|D|E|F|G|H|I|J|K|L|M) ;;' in DRIVE
    assert "sc2d_box_k.sh sc2d_reduce.py sc2e_box_l.sh" in DRIVE and "; do STAGE=" in DRIVE and 'STAGE="$STAGE $P98/p98_bake.py"' in DRIVE
    assert 'sc2c_*|sc2d_*|sc2e_*) src="$SC2/$name";;' in DRIVE and 'p98_bake.py) src="$P98/$name";;' in DRIVE
    pin = (REPO / "bench" / "sc1" / "staged.sha256").read_text()
    for name in ("sc2d_box_k.sh", "sc2d_reduce.py", "p98_bake.py"):
        assert re.search(rf"^[0-9a-f]{{64}}  {re.escape(name)}$", pin, re.M), name


def test_box_k_serves_qwen3_6_at_p98_s_pin_and_gpt_oss_on_sc2g_s_path():
    p98 = (REPO / "bench" / "p98" / "p98_run.sh").read_text()
    m = re.search(r"^MODEL=(\S+); REV=([0-9a-f]{40})", p98, re.M)
    assert m and f"QW36_MID={m.group(1)}; QW36_REV={m.group(2)}" in BOX
    assert 'k_model gptoss "$SC2G_MID" "$SC2G_REV" "$GA_GPTOSS" "$SC2G_E4B_ENV $FOLDS"' in BOX
    assert 'k_model qw36 "$QW36_MID" "$QW36_REV" "$QA36" ""' in BOX                   # the server's defaults
    # gpt-oss's read lands before Qwen3.6 is fetched; a Qwen3.6 failure is recorded, not a finish
    assert BOX.index("phase K1") < BOX.index('fetch qw36 "$QW36_MID"')
    assert 'its read is UNREAD"; rec 12; fi' in BOX
    # the knob is the only per-arm difference, and nothing from the environment leaks a route or graph setting
    start = BOX[BOX.index("k_server_start(){"):BOX.index("SRV_PID=$!", BOX.index("k_server_start(){"))]
    assert "$SC2D_KNOB=$K" in start and "-u E4B_PAGED_PREFILL_GRAPH -u E4B_PAGED_GRAPHS" in start


def test_the_proof_runs_box_k_s_flow_and_the_reducer_s_proof_mode():
    proof = BOX[BOX.index("prove_k(){"):]
    assert 'k_model granite "$GR" "$GR_REV" "$W/work_granite/nf4.arena" "$GR_ENV"' in proof
    assert '"$PY" $W/sc2d_reduce.py --dir $W/sc2 --proof granite' in proof
    for s in ("sc2_driver.py", "sc2_identity.py", "sc2d_reduce.py"):
        assert s in proof
