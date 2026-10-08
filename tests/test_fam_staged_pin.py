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
    for script, want in (("fam_reduce.py", "fam_reduce self-test OK (39/39 cases)"),
                         ("fam_box.py", "fam_box self-test OK (26/26 cases)")):
        out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True, env=base)
        assert out.returncode == 0 and want in out.stdout, out.stdout + out.stderr
    assert "self-tested on 39 cases" in PREREG


def test_the_rule_is_the_registered_rule():
    r = _load("fam_reduce")
    assert (r.BIAS_MARGIN, r.SPREAD_MULT, r.SPREAD_MIN, r.AGREE_MARGIN) == (0.010, 2.0, 0.005, 0.005)
    assert (r.BACKSTOP_BIAS, r.BACKSTOP_AGREE) == (0.020, 0.90) and r.PHASE_C_ANCHOR_AGREE == 0.924
    for s in ("B_f + **0.010** nats", "2** × max(S_f, **0.005**)", "A_f − **0.005**", "**0.020** nats", "agree ≥ **0.90**"):
        assert s in PREREG, s
    assert (r.CONT, r.PROOF["cont"], r.WINDOWS_PER_SET) == (128, 32, 12)
    assert r.FAMILY_CONFIGS == {"granite": ("OFF", "ON_auto"), "gptoss": ("OFF", "ON_glue", "ON_r2", "ON_epi", "ON_auto"),
                                "qw36": ("OFF", "ON_auto")}
    assert r.PROOF["configs"] == {"granite": ("OFF", "ON_epi", "ON_auto")} and r.PROOF["anchor_family"] == "granite"
    assert r.ATTN_LAYERS == {"granite": 32, "gptoss": 24, "qw36": 10}
    assert r.EXTRA_DECODE_FORWARDS == {"granite": 0, "gptoss": 0, "qw36": 1}

    def text(c):
        return f"`{c[0]} / {c[1]} / [{c[2][0]}, {c[2][1]}] / {c[3]}`"
    for (fam, config), c in r.CENSUS.items():
        if config not in ("OFF", "ON_epi") or fam == "gptoss":
            if config != "OFF":
                assert text(c) in PREREG, (fam, config, text(c))
    rows = {"granite": "Granite", "gptoss": "gpt-oss", "qw36": "Qwen3.6"}
    for (fam, config), step in r.PER_STEP.items():
        if (fam, config) == ("granite", "ON_epi"):
            continue                                             # the proof's config: not in the reading's table
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
    assert 'TAGS="granite gptoss qw36"' in RUN and 'TAGS="granite"' in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert 'md.version("grouped-nf4-gemm") == "0.43.0"' in trip and 'transformers.__version__ == "5.17.0"' in trip
    assert '"n_split" in inspect.signature(fp8_paged_attn.fp8_paged_decode_attention).parameters' in trip


def test_every_time_left_check_fits_inside_its_guard():
    """The reading's guard is 4.5 h; the proof's 0.75 h. Every per-step need, plus the 600 s fetch-back, fits."""
    reading = re.search(r'TAGS="granite gptoss qw36"; CONT_DEF=128; (.*)', RUN).group(1)
    proof = re.search(r'TAGS="granite"; CONT_DEF=32; (.*)', RUN).group(1)
    for line, guard in ((reading, 4.5 * 3600), (proof, 0.75 * 3600)):
        vals = dict(re.findall(r"(NEED_\w+|CAP_\w+)=(\d+)", line))
        for k, v in vals.items():
            if k.startswith("NEED_"):
                assert int(v) + 600 <= guard, (k, v, guard)
    assert "guard 4.5 h" in PREREG and "guard 0.75 h" in PREREG
