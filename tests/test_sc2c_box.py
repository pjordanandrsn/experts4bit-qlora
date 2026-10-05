# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC2c's box H (#846): the checks the box runs on a server's own /health, executed rather than grepped.

``routes_ok`` and ``h_early`` are Python heredocs inside ``bench/sc2/sc2c_box_h.sh``. These tests cut each heredoc out
and run it on /health fixtures:
* the reading's Qwen3 int4 store must have TAKEN K19 above 256 rows (``prefill_routes.seen``);
* the proof's Granite NF4 store takes NF4's grouped route, so the proof only asks that a grouped call above 256 rows
  ran. That case once STOPped the proof before it ran: the seen check was the reading's alone;
* resolved-but-not-taken (the gpt-oss lesson) STOPs;
* the early engagement reads both knobs as registered.
"""
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
BOX = (REPO / "bench" / "sc2" / "sc2c_box_h.sh").read_text()


def _heredoc(fn: str, tag: str) -> str:
    start = BOX.index(fn + "(){")
    body = BOX[BOX.index(f"<<'{tag}'", start):]
    body = body[body.index("\n") + 1:]
    return body[:body.index(f"\n{tag}\n")]


ROUTES_PY = _heredoc("routes_ok", "PYR")
EARLY_PY = _heredoc("h_early", "PYE")
ROUTES = {"int4_prefill_env": None, "prefill_attn_env": None, "device_grouping": True, "int4_prefill": "k19",
          "int4_prefill_above_256_rows": "k19", "prefill_attn": "flash"}
ENGINE = {"chunk_tokens": 512, "max_prefill_tokens_per_step": 512}


def _run(script, args, tmp_path):
    p = tmp_path / "s.py"
    p.write_text(script)
    return subprocess.run([sys.executable, str(p), *args], capture_output=True, text=True, timeout=60)


def _health(tmp_path, seen, **over):
    h = {"prefill_routes": dict(ROUTES, seen=seen), "engine": dict(ENGINE)}
    h.update(over)
    f = tmp_path / "h.json"
    f.write_text(json.dumps(h))
    return str(f)


QWEN_SEEN = {"moe": {"int4_k19|gt256": 192, "int4_gemv|le256": 1440}, "prefill_attn": {"flash": 192}}
GRANITE_SEEN = {"moe": {"nf4_mtile_captured|gt256": 160, "nf4_k25|le256": 1200}, "prefill_attn": {"flash": 160}}


def test_the_reading_requires_k19_taken_above_256_rows(tmp_path):
    assert _run(ROUTES_PY, [_health(tmp_path, QWEN_SEEN), "int4_k19|"], tmp_path).returncode == 0
    # resolved k19, but the forward ran M-tile: STOP
    mtile = {"moe": {"int4_mtile_captured|gt256": 192}, "prefill_attn": {"flash": 192}}
    r = _run(ROUTES_PY, [_health(tmp_path, mtile), "int4_k19|"], tmp_path)
    assert r.returncode == 1 and "seen.moe_gt256" in r.stdout


def test_the_proof_on_granites_nf4_store_passes_without_the_int4_prefix(tmp_path):
    assert _run(ROUTES_PY, [_health(tmp_path, GRANITE_SEEN), ""], tmp_path).returncode == 0
    # the same Granite health under the reading's prefix would STOP: the bug this guards against
    assert _run(ROUTES_PY, [_health(tmp_path, GRANITE_SEEN), "int4_k19|"], tmp_path).returncode == 1


def test_no_grouped_call_or_no_flash_stops_either_way(tmp_path):
    none = {"moe": {"int4_gemv|le256": 10}, "prefill_attn": {"flash": 4}}
    sinks = {"moe": {"int4_k19|gt256": 4}, "prefill_attn": {"explicit_mask:sinks": 4}}
    for seen in (none, sinks, {}):
        for prefix in ("", "int4_k19|"):
            assert _run(ROUTES_PY, [_health(tmp_path, seen), prefix], tmp_path).returncode == 1, (seen, prefix)
    # a resolved route off the registered one, or a chunk budget that splits a first chunk
    bad = _health(tmp_path, QWEN_SEEN, engine={"chunk_tokens": 512, "max_prefill_tokens_per_step": 1024})
    assert _run(ROUTES_PY, [bad, "int4_k19|"], tmp_path).returncode == 1


def _early(tmp_path, arm, kv, graph=None, rows=64, prompt_tokens=512):
    h = {"prefill_graph": graph or {"status": "on", "replays": 4, "eager_chunks": 0}, "kv_bookkeeping": kv}
    (tmp_path / "h.json").write_text(json.dumps(h))
    (tmp_path / "w.json").write_text(json.dumps({"requests": [{"prompt_tokens": prompt_tokens}] * 4}))
    (tmp_path / "s.jsonl").write_text("".join(json.dumps({"step": i}) + "\n" for i in range(rows)))
    return _run(EARLY_PY, [arm, str(tmp_path / "h.json"), str(tmp_path / "w.json"), str(tmp_path / "s.jsonl")], tmp_path)


OFF = {"requested": False, "bulk": False, "flush_layers": 4, "flush_bulk": 0, "ready_layers": 4, "ready_bulk": 0,
       "ready_at_flush": 0}
ON = {"requested": True, "bulk": True, "flush_layers": 0, "flush_bulk": 4, "ready_layers": 0, "ready_bulk": 0,
      "ready_at_flush": 4}


def test_early_engagement_reads_both_knobs_as_registered(tmp_path):
    assert _early(tmp_path, "off", OFF).returncode == 0
    assert _early(tmp_path, "on", ON).returncode == 0
    assert _early(tmp_path, "on", OFF).returncode == 1                       # knob not engaged
    assert _early(tmp_path, "off", ON).returncode == 1
    assert _early(tmp_path, "on", ON, graph={"status": "refused", "why": "memory"}).returncode == 1
    assert _early(tmp_path, "on", ON, rows=10).returncode == 1               # no step trace yet
    assert _early(tmp_path, "on", ON, prompt_tokens=511).returncode == 1


def test_box_h_is_wired_and_both_server_starts_name_their_seen_route():
    run = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
    assert "H) . $W/sc2_box_e.sh; . $W/sc2c_box_h.sh; install_sc2_client ;;" in run
    assert '[ "$BOX" = H ] && prove_h' in run and "H) box_h;;" in run
    assert 'case "$BOX" in H) GNF4_SHA=dc8f94abfd868f149178623f6eb403dc8b892b02;; esac' in run
    assert '"$SPEEDENV E4B_PAGED_FUSE_QKV=1" "int4_k19|"; then' in BOX               # the reading: K19 taken
    assert '"$W/work_granite/nf4.arena" "$GR_ENV" ""; then' in BOX                  # the proof: any grouped route
    assert "env -u E4B_INT4_PREFILL -u E4B_PAGED_PREFILL_ATTN -u E4B_PAGED_PREFILL_GRAPH" in BOX
    assert "E4B_PAGED_STEP_TRACE=$W/sc2/steps_$TAG.jsonl" in BOX and "$SC2C_KNOB=$K" in BOX


def test_every_python3_box_pins_torch_2_8_including_h():
    """SC2c A1: box H was left out of the torch pin when box I's harness merged beside it, and sc2c-prove-2 installed
    whatever torch pip resolved (2.14.1+cu130), not SC2b's and SC2g's 2.8.0+cu128. Every box that installs the e4b
    stack with python3 (C..I) pins it."""
    run = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
    line = next(ln for ln in run.splitlines() if 'pipx logs/pip_torch.log 1800 "torch==2.8.0"' in ln)
    for box in "CDEFGHI":
        assert f'[ "$BOX" = {box} ]' in line, box
