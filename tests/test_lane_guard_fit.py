"""Every rental runner that gates on time left refuses BEFORE its first install unless its guard fits its work with headroom.

The cross-lane rule (2026-10-19), after three lanes in one day sized a guard to fit exactly:
- P149 box K: NEED_BENCH 1200 s plus the 600 s reserve was the whole 0.5 h guard, so no box could reach its bench after
  any setup (p149k-5090-6, rc 40);
- smt-sib: a fit taken from one host's walls, and the next host slower;
- P154: a 5 % margin on one host's walls, caught in review.

A runner that is not on the frozen legacy list carries the canonical pre-install fit check (bench/common/README.md,
"Guard fit"): FIT = its install allowance + its largest NEED_* + RESERVE, and it refuses with its own exit code unless
the time left on the box is at least FIT x FIT_HEADROOM_PCT / 100, with FIT_HEADROOM_PCT at least 125. The check sits
before the first install, so a guard that cannot fit costs cents, not a run. It is not a strike, but two consecutive
headroom refusals on one lane count as one strike: the guard, not the hosts, is then wrong.

The static side stays per lane: each lane's own test asserts (boot + its declared fit) x 1.25 <= its registered guard
(as tests/test_p155.py does), boot being the time from the launch, where the deadline starts, to the check. This test
checks the structure only; it parses no PREREG.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_HEADROOM_PCT = 125
STRIKE_NOTE = "two consecutive headroom refusals count as one strike"
INSTALL_RE = re.compile(
    r"(?:\bpip3?|\bpython3?\s+-m\s+pip|\$\{?PIP\}?\"?|\b(?:micro)?mamba|\bconda|\buv\s+pip)\s+install\b"
    r"|\buv\s+(?:sync|add)\b|apt-get install|^\s*pipx\s", re.M)
# a script run from another file: its installs are invisible here, so the fit must come before it too
EXTERNAL_RE = re.compile(r"^\s*(?:\.|source)\s+\S|\b(?:bash|sh)\s+\S+\.sh\b|\bpython3?\s+(?!-)\S+\.py\b", re.M)

# Runners that gated on time left before this rule. Frozen: a new runner is not added here, it carries the check.
LEGACY_UNFITTED = frozenset([
    "fam/fam", "g4census/g4census", "gpu-contract/gc", "i8dense/i8", "k16/k16p5", "p100/p100", "p102/p102",
    "p103/p103", "p104/p104", "p105/p105", "p106/p106", "p107/p107", "p108/p108", "p109/p109", "p110/p110",
    "p111/p111", "p112/p112", "p113/p113", "p115/p115", "p115/p115c", "p115/p115d", "p116/p116", "p117/p117",
    "p118/p118", "p119/p119", "p120/p120", "p121/p121", "p122/p122", "p123/p123", "p124/p124", "p125/p125",
    "p126/p126", "p127/p127", "p130/p130", "p131/p131", "p132/p132", "p133/p133", "p134/p134", "p135/p135",
    "p136/p136", "p137/p137", "p140/p140", "p141/p141", "p143/p143", "p143/p143a2", "p143/p143a3", "p144/p144",
    "p145/p145", "p146/p146", "p148/p148", "p149/p149k", "p150/p150", "p151/p151", "p152/p152", "p153/p153", "p154/p154",
    "p39/p39", "p42/p42", "p44/p44a", "p44/p44b", "p47/p47", "p54/p54", "p55x/p55x", "p57/p57", "p58/p58", "p59/p59",
    "p60/p60", "p63/p63", "p64/p64", "p65/p65", "p66/p66", "p68/p68", "p70/p70", "p86/p86", "p87/p87", "p88/p88",
    "p89/p89", "p90/p90", "p91/p91", "p92/p92", "p93/p93", "p94/p94", "p95/p95", "p96/p96", "p97/p97", "p98/p98",
    "p99/p99", "pername/pername", "sc1/sc1", "sd1/sd1", "sd2/sd2", "tc1/tc1", "tp4/tp4", "vsplit/vsplit",
])


def _gated_runners():
    """Every runner that decides on time left: it defines can_run()."""
    return sorted(p for p in ROOT.glob("bench/*/*_run.sh") if re.search(r"^\s*can_run\(\)", p.read_text(), re.M))


def _name(p: Path) -> str:
    return f"{p.parent.name}/{p.name.removesuffix('_run.sh')}"


def _brace_depth(text: str) -> int:
    """Open { minus closed }, after dropping ${...} expansions: above zero means inside a function body or a group."""
    t = re.sub(r"\$\{[^}]*\}", "", _code(text))
    return t.count("{") - t.count("}")


def _code(text: str) -> str:
    """The text without quoted strings and comments, for the depth scans."""
    t = re.sub(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'', '""', text)
    return re.sub(r"(?:^|(?<=\s))#.*$", "", t, flags=re.M)


def _block_depth(text: str) -> int:
    """Open if/case/while/until/for minus closed fi/esac/done: above zero means inside a branch or a loop."""
    t = _code(text)
    opened = len(re.findall(r"(?:^|[;&|(]|\bthen|\bdo|\belse)\s*(?:if|case|while|until|for)\b", t, re.M))
    return opened - len(re.findall(r"\b(?:fi|esac|done)\b", t))


def fit_problems(text: str) -> list[str]:
    """Why a runner's text does not carry the canonical pre-install fit check (empty: it does)."""
    out = []
    m = re.search(r"^FIT_HEADROOM_PCT=(\d+)", text, re.M)
    if not m:
        return ["no FIT_HEADROOM_PCT"]
    if int(m.group(1)) < MIN_HEADROOM_PCT:
        out.append(f"FIT_HEADROOM_PCT {m.group(1)} < {MIN_HEADROOM_PCT}")
    fit = re.search(r"\bFIT=\$\(\(([^)]*)\)\)", text)
    if not fit:
        out.append("no FIT=$((...))")
    elif "NEED_" not in fit.group(1) or "RESERVE" not in fit.group(1) or "INSTALL" not in fit.group(1):
        out.append("FIT does not add an INSTALL* allowance, a NEED_* and the RESERVE (a pre-install check must count "
                   "the install still ahead of it)")
    if "FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100))" not in text:
        out.append("no FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100))")
    check = re.search(r'^.*\[ "\$LEFT" -ge "\$FIT_NEED" \].*$', text, re.M)
    if not check:
        out.append('no [ "$LEFT" -ge "$FIT_NEED" ] refusal')
    else:
        line = check.group(0)
        if "${FIT_NEED}" not in line or not re.search(r"\bfinish \d+|\bexit \d+", line):
            out.append("the refusal must print ${FIT_NEED} (the quantity it compares) and exit with its own code")
        if not line.startswith('[ "$LEFT"') or _brace_depth(text[:check.start()]) != 0 \
                or _block_depth(text[:check.start()]) != 0:
            out.append("the refusal is not at top level (indented, commented out, or inside a function body or a branch): it may "
                       "never run")
        install = INSTALL_RE.search(text)
        if not install:
            out.append("no install found (pip, pip3, conda, mamba, uv, apt-get or pipx): the order cannot be checked")
        elif install.start() < check.start():
            out.append("the fit is checked after the first install")
        external = EXTERNAL_RE.search(text)
        if external and external.start() < check.start():
            out.append("a script from another file runs before the fit check")
    if STRIKE_NOTE not in text:
        out.append(f"no '{STRIKE_NOTE}' note beside the refusal's exit code")
    return out


def test_every_gated_runner_fits_its_guard_with_headroom_or_is_legacy():
    runners = _gated_runners()
    assert len(runners) > len(LEGACY_UNFITTED) // 2          # the glob found the bench tree
    offenders = {_name(p): fit_problems(p.read_text()) for p in runners if _name(p) not in LEGACY_UNFITTED}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"runners without the pre-install fit check (bench/common/README.md, 'Guard fit'): {offenders}"


def test_the_legacy_list_has_no_stale_or_already_fitted_entries():
    names = {_name(p): p for p in _gated_runners()}
    assert LEGACY_UNFITTED <= set(names), sorted(LEGACY_UNFITTED - set(names))
    fitted = sorted(n for n in LEGACY_UNFITTED if not fit_problems(names[n].read_text()))
    assert not fitted, f"fitted runners still on the legacy list (remove them): {fitted}"


CANONICAL = """
FIT_HEADROOM_PCT=125   # the guard keeps 25 % headroom over the fit (cross-lane rule)
FIT=$((INSTALL_TO_GATE_S + NEED_BENCH + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((DEADLINE - $(date +%s)))
# rc 19: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; finish 19; }
pipx(){ python -m pip install "$@"; }
"""


def test_the_rule_accepts_the_canonical_check_and_rejects_its_mutants():
    assert fit_problems(CANONICAL) == []
    mutants = {
        "no headroom": CANONICAL.replace("FIT_HEADROOM_PCT=125", "FIT_HEADROOM_PCT=100"),
        "no reserve": CANONICAL.replace(" + RESERVE))", "))"),
        "no need": CANONICAL.replace("NEED_BENCH", "BENCH_S"),
        "compared to the bare fit": CANONICAL.replace('-ge "$FIT_NEED"', '-ge "$FIT"'),
        "a silent refusal": CANONICAL.replace("needs ${FIT_NEED}s", "needs more"),
        "no exit code": CANONICAL.replace("finish 19;", "true;"),
        "after the install": "pipx(){ python -m pip install \"$@\"; }\n" + CANONICAL.replace(
            'pipx(){ python -m pip install "$@"; }\n', ""),
        "no strike note": CANONICAL.replace("two consecutive headroom refusals count as one strike", "fine"),
        "inside an uncalled function": CANONICAL.replace('[ "$LEFT"', 'never_called(){\n[ "$LEFT"').replace(
            "finish 19; }\n", "finish 19; }\n}\n"),
        "indented in a branch": CANONICAL.replace('[ "$LEFT"', 'if false; then\n  [ "$LEFT"').replace(
            "finish 19; }\n", "finish 19; }\nfi\n"),
        "a pip3 install first": "pip3 install x\n" + CANONICAL,
        "a sourced helper first": ". ./setup_box.sh\n" + CANONICAL,
        "a called installer first": "bash $W/install.sh\n" + CANONICAL,
        "no install at all": CANONICAL.replace('pipx(){ python -m pip install "$@"; }\n', ""),
        "commented out": CANONICAL.replace('[ "$LEFT"', '# [ "$LEFT"'),
        "in an unindented if false": CANONICAL.replace('[ "$LEFT"', 'if false; then\n[ "$LEFT"').replace(
            "finish 19; }\n", "finish 19; }\nfi\n"),
        "in a one-line if false": CANONICAL.replace('[ "$LEFT"', 'if false; then [ "$LEFT"').replace(
            "finish 19; }\n", "finish 19; }; fi\n"),
        "no install allowance": CANONICAL.replace("INSTALL_TO_GATE_S + ", ""),
        "a micromamba install first": "micromamba install -y x\n" + CANONICAL,
        "a $PIP install first": '"$PIP" install x\n' + CANONICAL,
    }
    survived = [k for k, t in mutants.items() if not fit_problems(t)]
    assert not survived, survived


def test_p149k_amendment_3_fit_is_the_shape_without_headroom():
    """P149 box K's Amendment 3 check (LEFT >= FIT, no headroom) is why the rule exists: it stays legacy, and the rule
    rejects it."""
    text = (ROOT / "bench/p149/p149k_run.sh").read_text()
    assert "p149/p149k" in LEGACY_UNFITTED and "FIT=$((INSTALL_TO_BENCH + NEED_BENCH + RESERVE))" in text
    assert fit_problems(text)
