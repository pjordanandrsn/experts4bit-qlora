# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane FAM's staged-file pin must match the repo, and the runner, box, reducer and registration must agree
(bench/fam/PREREG-fam.md; e4b#1362).

``bench/fam/fam_drive.sh`` refuses to run when a staged file's sha256 differs from ``bench/fam/staged.sha256``; this
test runs the same comparison in CI, where it costs nothing. It also pins:
- P115's quality box, P110's, P108's and P97's boxes, P39's bake and calibration, P98's bake and the fusion-modes test
  at Phase C's registered bytes (``bench/p115/staged-c.sha256``);
- the order: refusals, install and tripwire, the self-tests, the premise on the card, then per model (Granite first):
  fetch, bake, OFF, the ON configs, the anchor; then the reducer;
- the configs' knobs, the SC2g path's levers and the model pins as the box, the runner and the reducer state them;
- the rule's constants, census and per-step tables as the registration states them;
- every time-left check inside its guard.
"""
import hashlib
import importlib.util
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "fam"
PIN = LANE / "staged.sha256"
SOURCES = {
    "fam_run.sh": LANE / "fam_run.sh",
    "fam_box.py": LANE / "fam_box.py",
    "fam_reduce.py": LANE / "fam_reduce.py",
    "fam_speed.py": LANE / "fam_speed.py",
    "fam_speed_reduce.py": LANE / "fam_speed_reduce.py",
    "p115_quality.py": REPO / "bench" / "p115" / "p115_quality.py",
    "p110_box.py": REPO / "bench" / "p110" / "p110_box.py",
    "p108_box.py": REPO / "bench" / "p108" / "p108_box.py",
    "p97_box.py": REPO / "bench" / "p97" / "p97_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "p98_bake.py": REPO / "bench" / "p98" / "p98_bake.py",
    "test_fam_split1_gpu.py": REPO / "tests" / "test_fam_split1_gpu.py",
    "test_fusion_modes.py": REPO / "tests" / "test_fusion_modes.py",
}
PREMISE = ("test_fam_split1_gpu.py", "test_fusion_modes.py")
RUN = (LANE / "fam_run.sh").read_text()
DRIVE = (LANE / "fam_drive.sh").read_text()
PREREG = (LANE / "PREREG-fam.md").read_text(encoding="utf-8")


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _load(name):
    sys.path.insert(0, str(LANE))
    spec = importlib.util.spec_from_file_location(name, LANE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next FAM launch refuses ON A RENTED BOX"


def test_borrowed_files_run_at_phase_c_s_registered_bytes():
    mine = dict((n, w) for w, n in _entries())
    c = dict((n, w) for w, n in _entries(REPO / "bench" / "p115" / "staged-c.sha256"))
    for name in ("p115_quality.py", "p110_box.py", "p108_box.py", "p97_box.py", "k8_bake.py", "calib.json", "p98_bake.py",
                 "test_fusion_modes.py"):
        assert mine[name] == c[name], f"{name} is not at Phase C's bytes"


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "fam_run.sh":
            assert name in staged, name
    assert "--include 'work_*/bake.json' --exclude 'work_*/*'" in DRIVE, "arenas, snapshots and references stay on the box"


def test_the_self_tests_pass():
    base = {"PATH": "/usr/bin:/bin", **{k: os.environ[k] for k in ("SYSTEMROOT",) if k in os.environ}}
    for script, want in (("fam_reduce.py", "fam_reduce self-test OK (70/70 cases)"),
                         ("fam_box.py", "fam_box self-test OK (38/38 cases)"),
                         ("fam_speed.py", "fam_speed self-test OK (9/9 cases)"),
                         ("fam_speed_reduce.py", "fam_speed_reduce self-test OK (30/30 cases)")):
        out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True, env=base)
        assert out.returncode == 0 and want in out.stdout, out.stdout + out.stderr
    assert "self-tested on 42 cases" in PREREG and "self-test now runs 45 cases" in PREREG     # Amendment 1
    assert "box's self-test now runs 28 cases" in PREREG                                   # Amendment 2
    assert "reducer's self-test now runs 50 cases" in PREREG                               # Amendment 3
    assert "reducer's self-test now runs 56 cases" in PREREG                               # Amendment 5
    assert "reducer's self-test now runs 58 cases" in PREREG                               # Amendment 6
    assert "reducer's self-test\nnow runs 59 cases" in PREREG                              # Amendment 7
    flat = " ".join(PREREG[PREREG.index("## Amendment 8"):].split())
    assert "box's self-test now runs 38 cases" in flat and "reducer's self-test now runs 68 cases" in flat  # Amendment 8


def test_the_rule_is_the_registered_rule():
    r = _load("fam_reduce")
    assert (r.BIAS_MARGIN, r.SPREAD_MULT, r.SPREAD_MIN, r.AGREE_MARGIN) == (0.010, 2.0, 0.005, 0.005)
    assert (r.BACKSTOP_BIAS, r.BACKSTOP_AGREE) == (0.020, 0.90) and r.PHASE_C_ANCHOR_AGREE == 0.924
    for s in ("B_f + **0.010** nats", "2** × max(S_f, **0.005**)", "A_f − **0.005**", "**0.020** nats", "agree ≥ **0.90**"):
        assert s in PREREG, s
    assert (r.CONT, r.PROOF["cont"], r.WINDOWS_PER_SET) == (128, 32, 12)
    assert r.FAMILY_CONFIGS == {"granite": ("OFF", "ON_auto"), "gptoss": ("OFF", "ON_glue", "ON_r2", "ON_epi", "ON_auto"),
                                "qw36": ("OFF", "ON_auto"),
                                "mixtral": ("OFF", "ON_glue", "ON_r2", "ON_epi", "ON_auto"),       # Amendment 5
                                "gemma4": ("OFF", "ON_glue", "ON_epi", "ON_auto")}                 # Amendment 6
    assert r.PROOF["configs"] == {"granite": ("OFF", "ON_epi", "ON_auto"), "mixtral": ("OFF", "ON_epi", "ON_auto"),
                                  "gemma4": ("OFF", "ON_epi", "ON_auto")}
    assert r.PROOF["anchor_family"] == "granite"
    assert r.ATTN_LAYERS == {"granite": 32, "gptoss": 24, "qw36": 10, "mixtral": 32, "gemma4": 30}
    assert r.WARMUP_FORWARDS == {"granite": 0, "gptoss": 0, "qw36": 1, "mixtral": 0, "gemma4": 0}
    assert r.FIRST_CELL == "wikitext|12|A"

    def text(c):
        return f"`{c[0]} / {c[1]} / [{c[2][0]}, {c[2][1]}] / {c[3]}`"
    for (fam, config), c in r.CENSUS.items():
        if config not in ("OFF", "ON_epi") or fam == "gptoss":
            if config != "OFF":
                assert text(c) in PREREG, (fam, config, text(c))
    rows = {"granite": "Granite", "gptoss": "gpt-oss", "qw36": "Qwen3.6"}
    for (fam, config), step in r.PER_STEP.items():
        if (fam, config) == ("granite", "ON_epi") or fam in ("mixtral", "gemma4"):
            continue                    # the proof's config; Mixtral's rows are Amendment 5's table (its own test)
        line = next(x for x in PREREG.splitlines() if x.startswith(f"| {rows[fam]} {config} |"))
        cells = [x.strip() for x in line.strip("|").split("|")][1:]
        names = ("rmsnorm_rows", "rmsnorm_resid_rows", "scaled_resid_add_rows", "rope_heads", "router_epilogue")
        assert {n: int(v) for n, v in zip(names, cells) if v} == step, (fam, config, cells)
    box = _load("fam_box")
    assert r.CENSUS_KEYS == box.CENSUS_KEYS and r.FLOOR == ("rep", "chunk", "half", "split1")
    for tag, (model, rev) in r.MODELS.items():
        assert f'{tag}) echo "{model} {rev}"' in RUN, tag
        assert f"`{model}` @ `{rev}`" in PREREG, tag
    gnf4 = re.search(r"GNF4_SHA=([0-9a-f]{40})", RUN).group(1)
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert f"grouped-nf4-gemm.git@{gnf4}" in ci or gnf4 == "6ee2e10408161a9d3c874975c9191a7f2957e6f4", \
        "the runner's grouped-nf4-gemm pin is e4b CI's pin at registration"


def test_the_configs_and_the_sc2g_path_agree_across_box_and_runner():
    box = _load("fam_box")
    sc2g = re.search(r'SC2G_E4B_ENV="([^"]+)"', (REPO / "bench" / "sc2" / "sc2g_box_g.sh").read_text()).group(1)
    assert dict(x.split("=") for x in sc2g.split()) == box.SC2G_ENV, "the anchor runs SC2g's e4b path exactly"
    assert f'SC2G="{sc2g}"' in RUN
    knobs = RUN[RUN.index("knobs_of(){"):RUN.index("SC2G=")]
    want = {"OFF": dict.fromkeys(box.KNOBS, "0")}
    for config, kn in box.CONFIGS.items():
        assert kn == {**dict.fromkeys(box.KNOBS, "0"), **{k: v for k, v in kn.items() if v != "0"}}
    assert 'ON_glue) g=auto;; ON_r2) r=auto;; ON_epi) e=auto;; ON_auto) q=auto; g=auto; r=auto; e=auto;;' in knobs
    assert 'echo "E4B_PAGED_FUSE_QKV=$q E4B_FUSE_T1_GLUE=$g E4B_FUSE_T1_GLUE_R2=$r E4B_FUSE_ROUTER_EPI=$e"' in knobs
    assert want["OFF"] == box.CONFIGS["OFF"]
    assert "E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=16" in RUN, "the instrument builds the default server eager at 16 slots"
    assert "--tag anchor --texts wikitext --shapes 12 --sets A" in RUN


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["CUDA_PROBE=$(python -", "REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @',
             "python - <<'PYT'", "fam_reduce.py --self-test", "fam_box.py --self-test", "p115_quality.py --self-test",
             "python -m pytest " + " ".join(PREMISE), 'echo "premise ok"', "for M in $TAGS; do",
             'say "fetch $M $MODEL @ $REV"', "python $W/p98_bake.py", "python $W/k8_bake.py",
             "box $M OFF default", "box $M $C default", "box $M OFF sc2g", "box $M ON_auto sc2g",
             "python $W/fam_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'TAGS="$FAMILY"' in RUN and 'TAGS="granite"' in RUN, "Amendment 1: one family per box; the proof is Granite's"
    assert 'PF="--families $TAGS"' in RUN, "the box reduces its own family"
    assert "for v in FAM_PROVE FAM_FAMILY FAM_SPEED; do" in DRIVE and 'a reading names its family (FAM_FAMILY)' in DRIVE
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert 'md.version("grouped-nf4-gemm") == "0.43.0"' in trip and 'transformers.__version__ == "5.17.0"' in trip
    assert '"n_split" in inspect.signature(fp8_paged_attn.fp8_paged_decode_attention).parameters' in trip


def test_every_time_left_check_fits_inside_its_guard():
    """Amendment 1's guards: Granite 2.5 h, gpt-oss 3.75 h, Qwen3.6 4.0 h, the proof 1.25 h. Every per-step need, plus
    the 600 s fetch-back, fits; and the runner's table is the amendment's."""
    guards = {"granite": 2.5, "gptoss": 3.75, "qw36": 4.0, "mixtral": 6.0, "gemma4": 4.0}     # Mixtral: Amendment 8
    for fam, hours in guards.items():
        line = re.search(rf"^\s*{fam}\)\s+(NEED_FETCH=.*);;$", RUN, re.M).group(1)
        vals = dict(re.findall(r"(NEED_\w+|CAP_\w+)=(\d+)", line))
        assert set(vals) == {"NEED_FETCH", "NEED_BAKE", "NEED_OFF", "NEED_ON", "CAP_OFF", "CAP_ON"}, fam
        for k, v in vals.items():
            if k.startswith("NEED_"):
                assert int(v) + 600 <= hours * 3600, (fam, k, v)
        assert int(vals["CAP_OFF"]) >= int(vals["NEED_OFF"]) and int(vals["CAP_ON"]) >= int(vals["NEED_ON"]), fam
    for fam, hours in (("granite", 1.25), ("mixtral", 1.5), ("gemma4", 1.25)):  # the proofs (Amendments 5, 6)
        line = re.search(rf'^\s*{fam}\)\s+TAGS="{fam}"; (NEED_FETCH=.*);;$', RUN, re.M).group(1)
        proof = dict(re.findall(r"(NEED_\w+|CAP_\w+)=(\d+)", line))
        assert set(proof) == {"NEED_FETCH", "NEED_BAKE", "NEED_OFF", "NEED_ON", "CAP_OFF", "CAP_ON"}, fam
        for k, v in proof.items():
            if k.startswith("NEED_"):
                assert int(v) + 600 <= hours * 3600, (fam, k, v)
    amend = PREREG[PREREG.index("## Amendment 1"):]
    for s_ in ("| Granite | 2.5 h | fetch 900 s, bake 600 s, OFF 6300 / 7200 s, each ON 1100 / 1800 s |",
               "| gpt-oss (+ the anchor) | 3.75 h | fetch 1200 s, bake 900 s, OFF 6300 / 7200 s, each ON 1100 / 1800 s |",
               "| Qwen3.6 | 4.0 h | fetch 2400 s, bake 900 s, OFF 9500 / 10800 s, each ON 1600 / 2700 s |",
               "| the proof | 1.25 h | fetch 300 s, bake 300 s, OFF 1800 / 2400 s, each ON 400 / 900 s |",
               "**lane ceiling $18.00**"):
        assert s_ in amend, s_


def test_the_speed_read_is_amendment_4_s():
    """Amendment 4: FAM_SPEED=1 is Qwen3.6's alone; the box's knobs, the rule's bars, the guards (reading 1.5 h, proof
    1.0 h) and the order -- the self-tests before the premise, the speed process after the bake, its own reducer -- are
    the registration's."""
    sp, red = _load("fam_speed"), _load("fam_speed_reduce")
    assert sp.KNOBS == {"E4B_PAGED_FUSE_QKV": "0", "E4B_FUSE_T1_GLUE": "0", "E4B_FUSE_T1_GLUE_R2": "0",
                        "E4B_FUSE_ROUTER_EPI": "1"}
    assert ("E4B_PAGED_FUSE_QKV=0 E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=1 \\\n"
            "      E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=2") in RUN
    assert (red.FAST, red.SLOW, red.NOISE, red.CENSUS, red.ROUTERS) == (0.98, 1.02, 0.015, [0, 0, [0, 0], 40], 40)
    assert (sp.WARM, sp.STEPS, sp.BUSY, sp.PROOF_STEPS, sp.SLOT, sp.KV_BATCH) == (5, 256, 32, 32, {"OFF": 0, "ON": 1}, 2)
    assert 'SPEED_STEPS=256; [ "$PROVE" = 1 ] && SPEED_STEPS=32' in RUN
    assert "refusing: the speed read is Qwen3.6's" in RUN and "refusing: FAM_SPEED=1 is Qwen3.6's" in DRIVE
    line = re.search(r'TAGS="qw36"; CONT_DEF=128; (.*)', RUN).group(1)
    vals = dict(re.findall(r"(NEED_\w+|CAP_\w+)=(\d+)", line))
    assert set(vals) == {"NEED_FETCH", "NEED_BAKE", "NEED_SPEED", "CAP_SPEED"} and int(vals["CAP_SPEED"]) >= int(vals["NEED_SPEED"])
    for hours in (1.5, 1.0):                                    # the reading, the proof
        for k, v in vals.items():
            if k.startswith("NEED_"):
                assert int(v) + 600 <= hours * 3600, (hours, k, v)
    order = ["fam_box.py --self-test", "fam_speed.py --self-test", "fam_speed_reduce.py --self-test",
             "python -m pytest " + " ".join(PREMISE), "python $W/p98_bake.py", "python $W/fam_speed.py --out",
             "box $M OFF default", "python $W/fam_speed_reduce.py --rec", "python $W/fam_reduce.py --dir $W"]
    at = [RUN.index(x) for x in order]
    assert at == sorted(at), list(zip(order, at))
    amend = PREREG[PREREG.index("## Amendment 4"):]
    for s_ in ("| the reading | 1.5 h | fetch 2400 s, bake 900 s, speed 1200 / 1800 s |",
               "| the proof | 1.0 h | fetch 2400 s, bake 900 s, speed 1200 / 1800 s |",
               "both ratios ≤ **0.98**", "both ≥ **1.02**", "more than **1.5 %**",
               "speed reducer's self-test runs 30 cases", "speed box's self-test runs 9 cases"):
        assert s_ in amend, s_


def test_mixtral_is_amendment_5_s():
    """Amendment 5: Mixtral's pin, configs and tables are the registration's; the box refuses below its disk floor at the
    start and again before the fetch; the anchor stays out of Mixtral's proof; the driver accepts the family."""
    r = _load("fam_reduce")
    assert r.MODELS["mixtral"] == ("mistralai/Mixtral-8x7B-Instruct-v0.1", "eba92302a2861cdc0098cc54bc9f17cb2c47eb61")
    assert '`mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a2861cdc0098cc54bc9f17cb2c47eb61`' in PREREG
    assert 'mixtral) echo "mistralai/Mixtral-8x7B-Instruct-v0.1 eba92302a2861cdc0098cc54bc9f17cb2c47eb61"' in RUN
    assert r.FAMILY_CONFIGS["mixtral"] == ("OFF", "ON_glue", "ON_r2", "ON_epi", "ON_auto")
    assert r.PROOF["configs"]["mixtral"] == ("OFF", "ON_epi", "ON_auto") and r.FP32_ROUTERS == {"mixtral": 32}
    assert (r.ATTN_LAYERS["mixtral"], r.WARMUP_FORWARDS["mixtral"]) == (32, 0)
    amend = PREREG[PREREG.index("## Amendment 5"):]
    for config in r.FAMILY_CONFIGS["mixtral"][1:]:
        c = r.CENSUS[("mixtral", config)]
        step = ", ".join(f"`{n}` {v}" for n, v in r.PER_STEP[("mixtral", config)].items())
        assert f"| {config} | `{c[0]} / {c[1]} / [{c[2][0]}, {c[2][1]}] / {c[3]}` | {step} |" in amend, (config, step)
    assert "fp32_upstream" in amend
    assert 'DISK_DEF=220; FETCH_DISK_GB=165' in RUN and '"$MIN_DISK_GB" != "$DISK_DEF"' in RUN
    assert RUN.index("FETCH_DISK_GB\" ] ||") < RUN.index('say "fetch $M $MODEL @ $REV"')
    assert '{ [ "$PROVE" = 1 ] && [ "$M" = granite ]; }' in RUN and 'PF="--proof --families $TAGS"' in RUN
    assert "qw36|mixtral|gemma4) ;;" in DRIVE
    for s_ in ("| Mixtral | 4.0 h | fetch 3000 s, bake 1500 s, OFF 9500 / 10800 s, each ON 1600 / 2700 s |",
               "| Mixtral's proof | 1.5 h | fetch 3000 s, bake 1500 s, OFF 2400 / 3000 s, each ON 600 / 900 s |",
               "**lane ceiling $26.00**"):
        assert s_ in amend, s_


def test_gemma4_is_amendment_6_s():
    """Amendment 6: Gemma-4's pin, configs (no r2 arm) and tables are the registration's; its fetch floor; the cannot-say
    on the sliding window."""
    r = _load("fam_reduce")
    assert r.MODELS["gemma4"] == ("google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52")
    assert 'gemma4) echo "google/gemma-4-26B-A4B-it 4d7ae4984b7db7de8f8457170b3f1a419ee76d52"' in RUN
    assert 'gemma4) echo "OFF ON_glue ON_epi ON_auto"' in RUN
    assert (r.ATTN_LAYERS["gemma4"], r.WARMUP_FORWARDS["gemma4"]) == (30, 0) and "gemma4" not in r.FP32_ROUTERS
    amend = PREREG[PREREG.index("## Amendment 6"):]
    assert '`google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`' in amend
    for config in r.FAMILY_CONFIGS["gemma4"][1:]:
        c = r.CENSUS[("gemma4", config)]
        step = ", ".join(f"`{n}` {v}" for n, v in r.PER_STEP[("gemma4", config)].items())
        assert f"| {config} | `{c[0]} / {c[1]} / [{c[2][0]}, {c[2][1]}] / {c[3]}` | {step} |" in amend, (config, step)
    assert '[ "$TAGS" = gemma4 ] && FETCH_DISK_GB=120' in RUN
    for s_ in ("| Gemma-4 | 4.0 h | fetch 2400 s, bake 900 s, OFF 9500 / 10800 s, each ON 1600 / 2700 s |",
               "| Gemma-4's proof | 1.25 h | fetch 2400 s, bake 900 s, OFF 2400 / 3000 s, each ON 600 / 900 s |",
               "sliding window never binds"):
        assert s_ in amend, s_


def test_mixtral_s_server_pool_is_amendment_7_s():
    """Amendment 7: after fam-mixtral-prove-1 ran out of GPU memory building the server's own KV pool (16 slots x 4096
    tokens beside 26.8 GiB of NF4 weights), Mixtral's processes build it at 768 tokens a slot -- enough for the
    instrument's prompt, positions and margin -- and Mixtral's proof is guarded 2.0 h, sized from that run."""
    r = _load("fam_reduce")
    assert 'build_env_of(){ case $1 in mixtral) echo "E4B_PAGED_MAX_TOKENS_PER_SEQ=768";; esac; }' in RUN
    assert "E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=16 $(build_env_of $tag) E4B_SHA=$E4B_SHA" in RUN
    assert 768 >= r.PREFILL[0] + r.CONT + 16 and r.SERVER_TOKENS == {"mixtral": 768}
    assert '"server": {"max_seqs": cfg.max_seqs, "max_tokens_per_seq": cfg.max_tokens_per_seq}' in (LANE / "fam_box.py").read_text()
    amend = PREREG[PREREG.index("## Amendment 7"):]
    assert "| Mixtral's proof | 2.0 h | fetch 3000 s, bake 1500 s, OFF 2400 / 3000 s, each ON 600 / 900 s |" in amend
    line = re.search(r'^\s*mixtral\)\s+TAGS="mixtral"; (NEED_FETCH=.*);;$', RUN, re.M).group(1)
    need = dict(re.findall(r"(NEED_\w+)=(\d+)", line))
    measured = 2336                                     # fam-mixtral-prove-1: launch to the bake's end, clock-read
    assert measured + int(need["NEED_OFF"]) + 600 <= 2.0 * 3600


def test_mixtral_s_gating_is_amendment_8_s():
    """Amendment 8: after fam-mixtral-prove-2's x0.90 passed the gate on c4val1, Mixtral's OFF process runs x0.90 and
    x0.80 on every set; the runner, the box and the reducer name the same rungs; the claimed resolution is the weakest
    rung that fails every gated cell; and the reading's 6.0 h guard holds the measured setup and the scaled steps."""
    box, r = _load("fam_box"), _load("fam_reduce")
    assert box.GRADED["mut080"] == 0.80 and box.GATING_MUTANT == "mut090" and box.LADDER == ("mut095", "mut098")
    assert r.GATING == {"mixtral": ("mut090", "mut080")} and r.gating("gemma4") == ("mut090",)
    spec = re.search(r'gating_of\(\)\{ case \$1 in mixtral\) echo "--gating ([\w,]+)";; esac; \}', RUN).group(1)
    assert box.gating_arms(spec) == r.GATING["mixtral"]
    assert "--cont $CONT $(gating_of $tag) \"$@\"" in RUN
    assert '"gating": list(gating),' in (LANE / "fam_box.py").read_text()
    amend = PREREG[PREREG.index("## Amendment 8"):]
    assert "store `ff615c5c`" in amend and "weakest rung that fails the gate in every gated cell" in amend
    assert "| Mixtral | 6.0 h | fetch 3000 s, bake 1500 s, OFF 10800 / 12600 s, each ON 2000 / 2700 s |" in amend
    line = re.search(r"^\s*mixtral\)\s+(NEED_FETCH=.*);;$", RUN, re.M).group(1)
    v = {k: int(x) for k, x in re.findall(r"(NEED_\w+|CAP_\w+)=(\d+)", line)}
    off, on = 1513 * 5.33 + 245 * 5.27, 261 * 6.7          # fam-mixtral-prove-2 x Granite's proof-to-reading ratios
    assert v["CAP_OFF"] >= v["NEED_OFF"] >= off and v["CAP_ON"] >= v["NEED_ON"] >= on
    setup = 2336                                            # Amendment 7's measured launch-to-bake
    assert setup + off + 3 * on + v["NEED_ON"] + 600 <= 6.0 * 3600         # the fourth ON still starts


def test_amendment_9_launches_past_1482():
    """Amendment 9: after fam-mixtral-3 hit #1477's served-path TypeError, the remaining runs launch past #1482; the rule,
    tables and pins are unchanged, and the measured per-step counts it states are Amendment 5's table at four layers."""
    r = _load("fam_reduce")
    flat = " ".join(PREREG[PREREG.index("## Amendment 9"):].split())
    for s_ in ("#1482 (`508cdd03`) fixed it", "`staged.sha256` is unchanged", "No second Mixtral proof",
               "at most $23.14 of $26.00", "`da17bbc5` and at `508cdd03`"):
        assert s_ in flat, s_
    big = r.PER_STEP[("mixtral", "ON_auto")]
    layers = r.ATTN_LAYERS["mixtral"]
    assert big == {"rmsnorm_rows": layers + 1, "rmsnorm_resid_rows": layers, "rope_heads": 2 * layers,
                   "router_epilogue": layers}
    small = 4
    assert (f"ON_auto reads `rmsnorm_rows` {small + 1}, `rmsnorm_resid_rows` {small}, `rope_heads` {2 * small} "
            f"and `router_epilogue` {small}") in flat


def test_gemma4_s_per_step_by_shape_is_amendment_10_s():
    """Amendment 10: after fam-gemma4-prove-1 read 216 glue norms a step at T == 12 against Amendment 6's 271, Gemma-4's
    ON_glue and ON_auto count per shape: the glue fold's 64-row bound sends the per-head q/k norms with more rows to
    their own torch forward. Every other family and T == 1 keep their tables."""
    r = _load("fam_reduce")
    assert r.PER_STEP_BY_SHAPE == {("gemma4", "ON_glue"): {12: {"rmsnorm_rows": 216}},
                                   ("gemma4", "ON_auto"): {12: {"rmsnorm_rows": 216, "router_epilogue": 30}}}
    for config in ("ON_glue", "ON_auto"):
        assert r.per_step("gemma4", config, 1) == r.PER_STEP[("gemma4", config)]
    assert r.per_step("gemma4", "OFF", 12) == {} and r.per_step("mixtral", "ON_auto", 12) == r.PER_STEP[("mixtral", "ON_auto")]
    flat = " ".join(PREREG[PREREG.index("## Amendment 10"):].split())
    for s_ in ("store `37e7f190`", "271 − (25 + 25 + 5) = 216", "reducer's self-test now runs 70 cases",
               "Amendment 6's 4.0 h guard stands"):
        assert s_ in flat, s_
