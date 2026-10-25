# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The G3 GPU check kit (``bench/kv-pool-g/g3-gpu-check/``), on the CPU.

- The verdict's self-test, and the verdict on /health blocks the runtime's OWN functions build: ``size_shared_pool`` on
  sc2j-5090-1's inputs (``tests/test_kv_pool_sizing.py``'s fixture), ``_kv_pool_health``, ``PagedModelRunner``'s
  ``prefill_graph_stats`` / ``note_prefill_graph_refused`` and ``prefill_graph_report``. So the field names the verdict
  binds on are the ones the server emits, and a captured graph reads PASS while the box's recorded refusal reads FAIL.
- The capture evidence the verdict cites: under ``auto`` the runner captures with ``require_headroom``, and
  ``self._prefill_graph`` is assigned only after the headroom gate.
- The guard fit: the canonical pre-install check (bench/common/README.md "Guard fit") sits before every install, and
  (boot + fit) x 1.25 fits the 1.0 h guard.
- The scripts' plumbing: the pin, the driver's refusals and the fetch (bench/common rule 8, also in
  ``tests/test_fetch_no_tar_fallback.py``).
"""
import importlib.util
import inspect
import os
import pathlib
import re
import subprocess
from types import SimpleNamespace

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
KIT = REPO / "bench" / "kv-pool-g" / "g3-gpu-check"
RUN = (KIT / "g3check_run.sh").read_text()
DRIVE = (KIT / "g3check_drive.sh").read_text()
_spec = importlib.util.spec_from_file_location("g3check", KIT / "g3check.py")
g3c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g3c)
GUARD_S = 3600          # the approved 1.0 h guard
BOOT_S = 300            # creation, boot, ssh wait and staging (sc2j-5090-1: 218 s)


def test_the_self_test_passes():
    assert g3c.self_test() == 0


# ---- /health from the runtime's own functions
def _runner(captured: bool, replays: int = 6):
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    r = SimpleNamespace(_prefill_graph=None, _pg_refused=None, _pg_stats=None)
    if captured:
        r._prefill_graph = {"T": 512, "pool_bytes": 426 * 2**20, "free_after_bytes": 1800 * 2**20,
                            "bulk_flush_bytes": 319 * 2**20}
        r._pg_stats = {"replays": replays, "eager_chunks": 6, "eager_reasons": {"later_chunk": 6, "short_chunk": 0}}
    else:
        PagedModelRunner.note_prefill_graph_refused(r, "memory: the graph's private pool holds 426 MiB for its life and "
                                                       "724 MiB is free after capture")
    r.prefill_graph_stats = lambda: PagedModelRunner.prefill_graph_stats(r)
    return r


def _health(captured=True, replays=6, requested="auto", kv_pool="shared", widths=None):
    import experts4bit_qlora.serve_recipe as sr
    from experts4bit_qlora.serve_paged import _kv_pool_health, prefill_graph_report
    from test_kv_pool_sizing import BOX, _box_setup, _qwen3_30b
    res = sr.size_shared_pool(_qwen3_30b(), _box_setup(), BOX["free"], widths=widths or sr.MAX_SEQS_AUTO_WIDTHS,
                              extra_bytes=BOX["extra"])
    runner = _runner(captured, replays)
    pick = next(c for c in res["candidates"] if c.get("fits"))
    runner.kv = SimpleNamespace(_n_rows=pick["pool_rows"], claim_all_rows=pick["claim_all_rows"], n_scratch=pick["scratch_slots"],
                                bt=16)
    parts = SimpleNamespace(runner=runner, scheduler=None)
    cfg = SimpleNamespace(kv_pool=kv_pool, kv_pool_resolution=res, prefill_graph=requested)
    # the configuration fields as sc2j-5090-1's own /health carried them (the fixture below)
    return {"model": "Qwen/Qwen3-30B-A3B",
            "levers": {"moe_layers": 48, "int4_expert_layers": 48, "int4_attn_projections": 96,
                       "attn_int4_rtn_projections": 192, "fuse_qkv_n": 48},
            "engine": {"max_seqs": res["max_seqs"], "max_tokens_per_seq": 4096, "kv_pool": kv_pool,
                       "kv_pool_sizing": _kv_pool_health(cfg, parts)},
            "prefill_graph": prefill_graph_report(cfg, SimpleNamespace(parts=parts))}


def test_a_captured_replayed_graph_on_the_box_inputs_reads_pass():
    v = g3c.verdict(_health())
    assert v["verdict"] == "PASS", v["why"]
    assert v["max_seqs"] == 64 and v["prefill_graph"]["status"] == "on"


def test_the_box_recorded_refusal_reads_fail():
    v = g3c.verdict(_health(captured=False))
    assert v["verdict"] == "FAIL" and any("refused" in w for w in v["why"])


def test_a_graph_that_never_replayed_reads_fail():
    v = g3c.verdict(_health(replays=0))
    assert v["verdict"] == "FAIL" and any("replays" in w for w in v["why"])


def test_a_forced_graph_reads_fail():
    """Under E4B_PAGED_PREFILL_GRAPH=1 the capture skips the headroom gate, so 'on' is no evidence of it."""
    assert g3c.verdict(_health(requested="1"))["verdict"] == "FAIL"


def test_claim_all_reads_fail():
    """Under claim-all _kv_pool_health is None: no sizing record, FAIL."""
    assert g3c.verdict(_health(kv_pool="claim-all"))["verdict"] == "FAIL"


def test_a_narrower_width_reads_fail():
    assert g3c.verdict(_health(widths=(32,)))["verdict"] == "FAIL"


def test_with_no_hold_back_the_candidate_reads_fail(monkeypatch):
    """The sizing without #2087's hold-back (a zero rule) leaves the 64-slot candidate's headroom at 0: FAIL even if a
    graph were on."""
    import experts4bit_qlora.serve_recipe as sr
    monkeypatch.setattr(sr, "prefill_graph_headroom_bytes", lambda pool, flush: 0)
    v = g3c.verdict(_health())
    assert v["verdict"] == "FAIL" and any("hold-back" in w for w in v["why"])


@pytest.mark.parametrize("name,path,value", [
    ("wrong model", ("model",), "ibm-granite/granite-3.1-3b-a800m-instruct"),
    ("2048 tokens a slot", ("engine", "max_tokens_per_seq"), 2048),
    ("bf16 experts", ("levers", "int4_expert_layers"), 0),
    ("bf16 attention", ("levers", "int4_attn_projections"), 0),
    ("int4 attention on 95 of 96 served projections", ("levers", "int4_attn_projections"), 95),
])
def test_each_configuration_mutant_reads_fail(name, path, value):
    h = _health()
    node = h
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    v = g3c.verdict(h)
    assert v["verdict"] == "FAIL" and len(v["why"]) == 1, (name, v["why"])


# ---- sc2j-5090-1's own /health (the shared arm's start record, receipt store sha256 below): the field names are the
# server's, and the record reads FAIL for the refusal and for the stack without #2087's hold-back
SC2J_HEALTH = REPO / "tests" / "fixtures" / "g3check" / "sc2j-5090-1_health_e4b_sh_d1_start.json"
SC2J_HEALTH_SHA256 = "87e43542a1207e39df0e74c0cb0f91809a2da51e263f7186eb489f6b957c83b8"


def _sc2j_health():
    import hashlib
    import json
    raw = SC2J_HEALTH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SC2J_HEALTH_SHA256
    return json.loads(raw)


def test_the_box_own_record_reads_fail_for_the_refusal_and_the_missing_hold_back():
    v = g3c.verdict(_sc2j_health())
    assert v["verdict"] == "FAIL"
    assert v["model"] == "Qwen/Qwen3-30B-A3B" and v["max_tokens_per_seq"] == 4096 and v["max_seqs"] == 64
    assert v["int4_expert_layers"] == v["moe_layers"] == 48 and v["int4_attn_projections"] == 96
    assert v["attn_int4_rtn_projections"] == 192 and v["fuse_qkv_n"] == 48
    assert sorted(w.split(" ")[0] for w in v["why"]) == sorted(
        ["no", "prefill_graph.status", "prefill_graph.T", "prefill_graph.replays"]), v["why"]
    assert any("prefill_graph_headroom_bytes" in w for w in v["why"])


def test_the_box_own_record_with_a_capture_and_the_hold_back_reads_pass():
    """Only the fields #2087 and a capture change: the 64-slot candidate's hold-back, and the graph's record."""
    h = _sc2j_health()
    cand = next(c for c in h["engine"]["kv_pool_sizing"]["derivation"]["candidates"] if c["max_seqs"] == 64)
    cand["prefill_graph_headroom_bytes"] = 1_929_379_840
    h["prefill_graph"] = {"status": "on", "T": 512, "replays": 6, "eager_chunks": 6, "pool_mib": 426,
                          "free_after_mib": 1800, "bulk_flush_mib": 319, "requested": "auto"}
    assert g3c.verdict(h)["verdict"] == "PASS", g3c.verdict(h)["why"]


# ---- the capture evidence the verdict cites
def test_on_under_auto_is_set_only_after_the_headroom_gate():
    from experts4bit_qlora import serve_paged
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    src = inspect.getsource(PagedModelRunner.enable_prefill_graph)
    gate = src.index('if require_headroom and pg["free_after_bytes"] < prefill_graph_headroom_bytes(')
    assert src.index("self._prefill_graph = pg") > gate
    assert src.count("self._prefill_graph = pg") == 1
    stats = inspect.getsource(PagedModelRunner.prefill_graph_stats)
    assert 'if pg is None:' in stats and '"status": "on"' in stats
    assert 'require_headroom=(cfg.prefill_graph == "auto")' in inspect.getsource(serve_paged)


# ---- the guard fit
def _int(name):
    m = re.search(rf"\b{name}=(\d+)\b", RUN)
    assert m, name
    return int(m.group(1))


def test_the_guard_fit_check_precedes_every_install():
    fit = RUN.index('[ "$LEFT" -ge "$FIT_NEED" ] ||')
    for pat in ("apt-get install", "pipx(){", "-m venv", "pip install"):
        assert RUN.index(pat) > fit, pat
    assert "finish 17" in RUN[fit:fit + 200] and "FIT_HEADROOM_PCT=125" in RUN
    assert "two consecutive headroom refusals count as one strike" in RUN


def test_boot_plus_fit_with_headroom_fits_the_guard():
    fit = _int("INSTALL_S") + _int("FETCH_BAKE_S") + _int("NEED_SERVER") + _int("RESERVE")
    assert re.search(r"INSTALL_TO_GATE_S=\$\(\(INSTALL_S \+ FETCH_BAKE_S\)\)", RUN)
    assert re.search(r"FIT=\$\(\(INSTALL_TO_GATE_S \+ NEED_SERVER \+ RESERVE\)\)", RUN)
    assert (BOOT_S + fit) * 125 // 100 <= GUARD_S, (BOOT_S + fit) * 1.25


def test_the_allowances_cover_the_measured_walls():
    """sc2j-5090-1 (same model, card and installs): installs 372 s, the bf16 fetch and the bake 1,184 s, the server's
    start with its capture about 140 s. This box adds the card burn (20 s plus the probe) before the e4b install."""
    assert _int("INSTALL_S") >= 372 + 60 and _int("FETCH_BAKE_S") >= 1184 and _int("NEED_SERVER") >= 140


# ---- the box and the controller
def test_the_pin_covers_every_staged_file_and_matches():
    out = subprocess.run(["bash", str(KIT / "make_pin.sh"), "--check"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    names = [ln.split()[1] for ln in (KIT / "staged.sha256").read_text().splitlines()]
    assert names == ["g3check_run.sh", "g3check.py", "k8_bake.py", "calib.json", "card_clock.py"]
    for n in names:
        assert n in RUN.split("for f in g3check_run.sh", 1)[1].split(";", 1)[0] or n == "g3check_run.sh"


def test_the_box_script_parses():
    assert subprocess.run(["bash", "-n", str(KIT / "g3check_run.sh")]).returncode == 0
    assert subprocess.run(["bash", "-n", str(KIT / "g3check_drive.sh")]).returncode == 0


def test_the_server_is_box_q_shared_arm():
    """The env box Q gave its shared arm (bench/sc2/sc2j_box_q.sh q_server_start + SC1's stack), at the prefill graph's
    default."""
    for frag in ("E4B_PAGED_MAX_SEQS=auto", "E4B_PAGED_KV_POOL=shared", "E4B_PAGED_MAX_TOKENS_PER_SEQ=4096",
                 "E4B_PAGED_FUSE_QKV=1", "-u E4B_PAGED_PREFILL_GRAPH",
                 'SPEEDENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 $FOLDS"',
                 'FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"',
                 'ROUTEENV="E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto"'):
        assert frag in RUN, frag
    sc1 = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
    for name in ("FOLDS", "SPEEDENV", "ROUTEENV"):
        line = next(ln for ln in RUN.splitlines() if ln.startswith(f"{name}="))
        assert line.split("#")[0].strip() in sc1, name
    assert "E4B_PAGED_PREFILL_GRAPH=" not in RUN.replace("-u E4B_PAGED_PREFILL_GRAPH", "")
    assert "QWEN_REV:-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in DRIVE and "MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in sc1


def test_the_card_clock_is_recorded_not_gated():
    line = next(ln for ln in RUN.splitlines() if "card_clock.py burn" in ln and not ln.startswith("#"))
    assert "--floor 2700" in line and "finish" not in line and "CC=$?" in line


def test_the_gnf4_default_is_main_ci_pin():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    sha = re.search(r"GNF4_SHA=\$\{GNF4_SHA:-([0-9a-f]{40})\}", DRIVE).group(1)
    assert f"grouped-nf4-gemm.git@{sha}" in ci


def test_the_driver_refuses_without_the_launcher(tmp_path):
    out = subprocess.run(["bash", str(KIT / "g3check_drive.sh")], capture_output=True, text=True,
                         env={"PATH": os.environ["PATH"]})
    assert out.returncode == 78 and "refusing" in out.stdout


def test_the_driver_binds_the_commit_under_test():
    assert 'G3_FIX=fc1f630a9672a7e034d0aa2ec13bae4e24ed8648' in DRIVE
    assert 'merge-base --is-ancestor "$G3_FIX" "$E4B_SHA"' in DRIVE
    assert 'merge-base --is-ancestor "$E4B_SHA" origin/main' in DRIVE
    assert DRIVE.index("is-ancestor") < DRIVE.index("$SSH ")


def test_the_driver_excludes_the_checkpoint_and_venv():
    assert "--exclude venv --exclude work_qwen3 --exclude .cache" in DRIVE


@pytest.mark.parametrize("bad", ["fc1f630a", "x" * 40])
def test_the_box_refuses_a_bad_sha(tmp_path, bad):
    env = {"PATH": os.environ["PATH"], "G3C_RUN_NONCE": "n", "G3C_RUN_ID": "r", "G3C_DEADLINE_EPOCH": "9999999999",
           "G3C_INSTANCE_ID": "1", "E4B_SHA": bad, "GNF4_SHA": "0" * 40, "QWEN_REV": "0" * 40}
    script = RUN.replace("W=/root/g3c", f"W={tmp_path}")
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert out.returncode == 78 and (tmp_path / "TP_DONE.n").exists() and not (tmp_path / "G3C_SUCCESS.n").exists()


# ---- the controller's verdict tail and the fetch, as bench/sc1's harness carries them (e4b#2041)
def _tail(tmp_path, rc, marker):
    """The driver's own block from the nonce check to its last line, against a fetched dir."""
    start = DRIVE.index('[ "$(cat "$RUN_DIR/g3c/G3C_RUN_NONCE"')
    block = DRIVE[start:]
    d = tmp_path / "run" / "g3c"
    d.mkdir(parents=True)
    (d / "G3C_RUN_NONCE").write_text("n\n")
    (d / "TP_DONE.n").write_text("")
    (d / "G3C_EXIT_CODE.n").write_text(f"{rc}\n")
    (d / "summary.txt").write_text("G3CHECK_VERDICT {}\n")
    if marker:
        (d / "G3C_SUCCESS.n").write_text("")
    script = 'say(){ echo "$*"; }\nset -uo pipefail\nRUN_DIR="$1"; NONCE=n\n' + block
    return subprocess.run(["bash", "-c", script, "drive", str(tmp_path / "run")], capture_output=True, text=True)


@pytest.mark.parametrize("rc,marker,want", [(0, True, 0), (0, False, 24), (1, False, 1), (1, True, 1), (23, False, 23)])
def test_pass_needs_rc_zero_and_the_success_marker(tmp_path, rc, marker, want):
    out = _tail(tmp_path, rc, marker)
    assert out.returncode == want, out.stdout + out.stderr
    assert ("(PASS)" in out.stdout) == (want == 0)


def test_the_fetch_carries_e4b_2041s_brotli_pin_and_one_retry():
    assert '"huggingface_hub>=0.23" "brotli>=1.2.0"' in RUN
    fetch = RUN[RUN.index('say "fetch $QWEN @ $QWEN_REV"'):RUN.index('line "FETCH $QWEN@$QWEN_REV')]
    assert fetch.count('"$PY" -c "$SNAP"') == 2 and "sleep 20" in fetch and ">> logs/fetch.log" in fetch
    assert fetch.count("alarm $(cap 2400)") == 2 and 'finish 11' in fetch


def test_the_summary_carries_no_gpu_uuid():
    line = next(ln for ln in RUN.splitlines() if ln.startswith("cat forensics.txt"))
    assert "cut -d, -f1-3,5- forensics.txt" in line and "uuid" in RUN.split("nvidia-smi --query-gpu=", 1)[1].split(" ", 1)[0]
