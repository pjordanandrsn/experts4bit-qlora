#!/usr/bin/env python3
"""sc2_kit_hold.py -- the SC2f/SC2h/SC2i kit hold at launch (SC2f Amendment 5, SC2h Amendment 3, SC2i Amendment 2).

Boxes N, O and P stage ONE flat kit (`bench/sc1/staged.sha256`, every entry on every box). Their kit was held at
`1a25b209`; the amendments move the hold to e4b#2041's merge, which carries the harness's brotli pin and fetch retry,
and declare everything else that merge changed in the kit as inert for boxes N, O and P. The stack under test is
unchanged (`E4B_SHA` = 36cdd2d4, set explicitly on the driver). This script is the check, run on the controller before
the create, its JSON recorded with the run:

  DELTA  (hold against BASE) the declared delta, exactly: the only existing entry whose bytes differ is `sc1_run.sh`;
         every one of its hunks is the brotli pip line, the fetch retry (two comment lines and two retry lines), a Q
         alternative added to a box list (each line with its Q removed is its BASE line), or one of box Q's two lines;
         and the entries added are exactly box Q's seven. No entry is removed. Every declared hunk is present, each once.
  PIN    (at the hold) every entry's sha256 is its source file's bytes at the hold commit.
  INERT  (at the hold) no staged entry other than the seven new files names one of them or box Q's functions, except
         `sc1_run.sh` on box Q's own three lines; in `sc1_drive.sh` and `make_pin.sh` the names appear only on staging
         lines (`STAGE=`, `src=`, `sha_of`), never on a line that runs anything. Every hit is recorded.
  HOLD   (at the launch commit, against the hold) every staged entry identical, name and sha256, and `sc1_drive.sh` and
         `make_pin.sh` byte-identical (they decide what is staged).

Each gate fails closed: an unreadable commit, file or pin, or a declared hunk that is absent, is a REFUSE.

  check --hold <rev> [--base <rev>] [--at <rev>] [--out FILE]     exit 0 OK, 1 REFUSE
  --self-test
"""
import argparse
import difflib
import hashlib
import json
import re
import subprocess
import sys

BASE = "1a25b209023c4621fffbf2ab399e2d282474df58"   # the kit hold SC2f Am. 4, SC2h Am. 2 and SC2i Am. 1 registered
HOLD = "7b8f1f69cf454433166cf349e7df6e7996349399"   # e4b#2041's merge commit, registered by the amendments
PIN = "bench/sc1/staged.sha256"
RUN = "sc1_run.sh"
DRIVERS = ("bench/sc1/sc1_drive.sh", "bench/sc1/make_pin.sh")
NEW = ("sc2j_box_q.sh", "sc2j_contract.py", "sc2j_prompts.py", "sc2j_reduce.py", "p117_reduce.py",
       "test_decode_graph_buckets.py", "g3_smoke.py")
TOKENS = re.compile(r"\b(sc2j_box_q|sc2j_contract|sc2j_prompts|sc2j_reduce|p117_reduce|test_decode_graph_buckets|g3_smoke"
                    r"|prove_q|box_q)\b")
BROTLI = ' "brotli>=1.2.0"'
FETCH = (
    '# huggingface_hub on httpx decodes brotli with an API that needs brotli >= 1.2.0 (pinned in the e4b install above); a',
    '# fetch that still fails is retried once after 20 s (the snapshot resumes) before it is a DL FAIL',
    '    || { say "fetch $NAME failed ($(tail -1 logs/fetch_$NAME.log | cut -c1-160)) -- one retry in 20 s"; sleep 20',
    '         perl -e "alarm $(arm_alarm $AL); exec @ARGV" "$PY" -c "from huggingface_hub import snapshot_download as s; print(s(\'$M\', revision=\'$R\', allow_patterns=[\'*.safetensors\',\'*.json\',\'tokenizer*\',\'*.model\',\'*.txt\',\'merges.txt\',\'vocab.json\'], max_workers=4))" >> logs/fetch_$NAME.log 2>&1; } \\',
)
Q_LINES = (
    '  Q) . $W/sc2_box_e.sh; . $W/sc2c_box_h.sh; . $W/sc2e_box_l.sh; . $W/sc2f_box_n.sh; . $W/sc2j_box_q.sh; install_sc2_client ;;   # SC2j: the shared KV pool against claim-all at auto width and 4,096 tokens a slot, e4b only (bench/sc2)',
    '  [ "$BOX" = Q ] && prove_q                                                   # SC2j: the contract arm, both pools\' servers, smokes on both ladders, QUALITY on 8 windows',
)
Q_STRIP = (('|Q)', ')'), (' || [ "$BOX" = Q ]', ''), (', "Q")', ')'), (' Q) PROVE_NEEDS="sc2client";;', ''),
           (' Q) box_q;;', ''), ('# P, Q: v0.50.0', '# P: v0.50.0'), ('O, P or Q (', 'O or P ('), (', SC2i, SC2j)', ', SC2i)'))
Q_REPLACED = 9                     # the box lists e4b#2041 extends with Q (sc1_run.sh at its head, c6e72f10)
STAGING = re.compile(r"STAGE=|\bsrc=|sha_of")
RUNS = re.compile(r"\bbash\b|\bsource\b|python|\$PY\b|\bexec\b")


def q_strip(line: str) -> str:
    for a, b in Q_STRIP:
        line = line.replace(a, b)
    return line


def resolve(name: str):
    """sc1_drive.sh's resolution case: a staged name -> its repo path (a wrong path fails the PIN gate)."""
    if name == "hook/usercustomize.py":
        return "bench/p42/hook/usercustomize.py"
    if "/" in name:
        return f"bench/sc1/{name}"
    if name.startswith(("sc1b_",)) or name == "kernel_classes.json":
        return f"bench/sc1b/{name}"
    if name.startswith(("sc2", "sc1g_")):
        return f"bench/sc2/{name}"
    if name.startswith("sc1_"):
        return f"bench/sc1/{name}"
    if name.startswith("test_"):
        return f"tests/{name}"
    if name == "p98_bake.py":
        return "bench/p98/p98_bake.py"
    if name in ("sc5_ref.json", "sc5_ref_chunked.json"):
        return f"bench/sc5/ref/{name}"
    if name.endswith(".lock.txt") or name == "e4b-wheels.lock":
        return f"bench/sc5/locks/{name}"
    if name.startswith("sc5_"):
        return f"bench/sc5/{name}"
    m = re.fullmatch(r"(p\d+)_(box|reduce)\.py", name)
    if m:
        return f"bench/{m.group(1)}/{name}"
    if name == "g3_smoke.py":
        return "bench/kv-pool-g/g3-smoke/g3_smoke.py"
    return f"bench/p39/{name}"


class Git:
    def __init__(self, repo="."):
        self.repo = repo

    def show(self, rev, path):
        r = subprocess.run(["git", "-C", self.repo, "show", f"{rev}:{path}"], capture_output=True)
        return r.stdout if r.returncode == 0 else None

    def sha(self, rev):
        r = subprocess.run(["git", "-C", self.repo, "rev-parse", "--verify", f"{rev}^{{commit}}"], capture_output=True,
                           text=True)
        return r.stdout.strip() if r.returncode == 0 else None


def entries(raw):
    """staged.sha256 -> {name: sha256}; None when absent, empty, malformed or holding a name twice."""
    if not raw:
        return None
    out = {}
    for line in raw.decode().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]) or parts[1].strip() in out:
            return None
        out[parts[1].strip()] = parts[0]
    return out or None


def classify_run(base: str, hold: str):
    """sc1_run.sh at BASE -> at the hold: (hunks, why). Every hunk must be declared, and every declared hunk present."""
    b, h = base.splitlines(), hold.splitlines()
    hunks, why = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, b, h, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        news = h[j1:j2]
        for new in news:
            if new in FETCH or new in Q_LINES:     # a declared insertion, wherever difflib grouped it
                hunks.append({"kind": "fetch-retry" if new in FETCH else "q-line", "line": new})
        news = [x for x in news if x not in FETCH and x not in Q_LINES]
        if op in ("replace", "insert") and len(news) == i2 - i1:
            for old, new in zip(b[i1:i2], news):
                if BROTLI in new and new.replace(BROTLI, "", 1) == old:
                    hunks.append({"kind": "brotli", "line": new})
                elif new != old and q_strip(new) == old:
                    hunks.append({"kind": "q-wiring", "line": new})
                else:
                    hunks.append({"kind": "UNDECLARED", "old": old, "line": new})
        else:
            hunks.append({"kind": "UNDECLARED", "op": op, "old": b[i1:i2], "line": news})
    if any(x["kind"] == "UNDECLARED" for x in hunks):
        why.append(f"{RUN}: {sum(x['kind'] == 'UNDECLARED' for x in hunks)} undeclared hunk(s)")
    n = {k: sum(x["kind"] == k for x in hunks) for k in ("brotli", "q-wiring")}
    if n["brotli"] != 1:
        why.append(f"{RUN}: the brotli pip line appears {n['brotli']} times, not once")
    if n["q-wiring"] != Q_REPLACED:
        why.append(f"{RUN}: {n['q-wiring']} Q-wiring lines, not {Q_REPLACED}")
    for line in FETCH + Q_LINES:
        c = sum(x["line"] == line for x in hunks)
        if c != 1:
            why.append(f"{RUN}: declared line inserted {c} times, not once: {line[:80]!r}")
    return hunks, why


def inert_hits(name: str, text: str, allowed: tuple):
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if TOKENS.search(line):
            hits.append({"file": name, "line_no": i, "text": line.strip()[:200], "allowed": line in allowed,
                         "staging": bool(STAGING.search(line)) and not RUNS.search(line)})
    return hits


def check(git, hold, base=BASE, at=None) -> dict:
    why, rec = [], {"base": base, "hold": hold, "at": at}
    if not hold:
        return {**rec, "verdict": "REFUSE", "why": ["no hold commit given (e4b#2041's merge)"]}
    for k in ("base", "hold", "at"):
        if rec[k] is not None:
            full = git.sha(rec[k])
            if not full:
                return {**rec, "verdict": "REFUSE", "why": [f"{k} {rec[k]} is not a readable commit"]}
            rec[k] = full
    base, hold, at = rec["base"], rec["hold"], rec["at"] or rec["hold"]
    rec["at"] = at
    eb, eh = entries(git.show(base, PIN)), entries(git.show(hold, PIN))
    if eb is None or eh is None:
        return {**rec, "verdict": "REFUSE", "why": [f"{PIN} unreadable at {'base' if eb is None else 'the hold'}"]}
    # DELTA
    changed = sorted(n for n in eb if n in eh and eb[n] != eh[n])
    added, removed = sorted(set(eh) - set(eb)), sorted(set(eb) - set(eh))
    rec.update(changed=changed, added=added, removed=removed)
    if changed != [RUN]:
        why.append(f"DELTA: changed entries {changed}, declared [{RUN!r}]")
    if added != sorted(NEW):
        why.append(f"DELTA: added entries {added}, declared {sorted(NEW)}")
    if removed:
        why.append(f"DELTA: removed entries {removed}")
    rb, rh = git.show(base, resolve(RUN)), git.show(hold, resolve(RUN))
    if rb is None or rh is None:
        why.append(f"DELTA: {RUN} unreadable")
        rec["run_hunks"] = None
    else:
        rec["run_hunks"], w = classify_run(rb.decode(), rh.decode())
        why += [f"DELTA: {x}" for x in w]
    # PIN
    bad = []
    for name, want in sorted(eh.items()):
        raw = git.show(hold, resolve(name))
        if raw is None or hashlib.sha256(raw).hexdigest() != want:
            bad.append(name)
    rec["pin_mismatch"] = bad
    if bad:
        why.append(f"PIN: {len(bad)} entries do not match their source at the hold: {bad[:5]}")
    # INERT
    hits = []
    for name in sorted(set(eh) - set(NEW)):
        raw = git.show(hold, resolve(name))
        if raw is None:
            continue                              # already a PIN refusal
        try:
            text = raw.decode()
        except UnicodeDecodeError:
            continue                              # binary entries cannot name a file
        hits += [{k: v for k, v in x.items() if k != "staging"} for x in inert_hits(name, text, Q_LINES + tuple(x["line"] for x in (rec["run_hunks"] or ())
                                                       if name == RUN and x["kind"] == "q-wiring"))]
    for path in DRIVERS:
        raw = git.show(hold, path)
        if raw is None:
            why.append(f"INERT: {path} unreadable at the hold")
            continue
        for x in inert_hits(path, raw.decode(), ()):
            x["allowed"] = x.pop("staging")
            hits.append(x)
    rec["inert_hits"] = hits
    if any(not x["allowed"] for x in hits):
        why.append("INERT: " + "; ".join(f"{x['file']}:{x['line_no']}" for x in hits if not x["allowed"]))
    # HOLD
    ea = entries(git.show(at, PIN))
    if ea is None:
        why.append(f"HOLD: {PIN} unreadable at the launch commit")
    else:
        diff = sorted(n for n in set(ea) | set(eh) if ea.get(n) != eh.get(n))
        rec["hold_diff"] = diff
        if diff:
            why.append(f"HOLD: entries differ from the hold: {diff}")
    for path in DRIVERS:
        a, h = git.show(at, path), git.show(hold, path)
        if a is None or a != h:
            why.append(f"HOLD: {path} differs from the hold (or is unreadable)")
    rec["verdict"], rec["why"] = ("OK" if not why else "REFUSE"), why
    return rec


# ------------------------------------------------------------------ self-test: each gate with its evidence removed
class Fake:
    def __init__(self, trees):
        self.trees = trees

    def show(self, rev, path):
        v = self.trees.get(rev, {}).get(path)
        return v.encode() if isinstance(v, str) else v

    def sha(self, rev):
        return rev if rev in self.trees else None


def _pin(tree):
    lines = ["# fixture pin"]
    for path, body in sorted(tree.items()):
        if path.startswith("bench/sc1/") and path.count("/") == 2 and not path.endswith((".sha256",)) \
                and path not in DRIVERS or path.startswith(("bench/sc2/", "bench/p117/", "tests/", "bench/kv-pool-g/")):
            lines.append(f"{hashlib.sha256(body.encode()).hexdigest()}  {path.rsplit('/', 1)[1]}")
    tree[PIN] = "\n".join(lines) + "\n"
    return tree


def _fixtures():
    base_run = "\n".join([
        'BOX=$SC1_BOX; case "$BOX" in N|O|P) ;; *) say "refusing: SC1_BOX must be N, O or P (SC2f, SC2h, SC2i)"; finish 78;; esac',
        'case "$BOX" in P) GNF4_SHA=41ef;; esac   # P: v0.50.0\'s commit',
        '  "huggingface_hub>=0.23" pytest || { say "PIP FAIL (e4b)"; finish 9; }',
        'fetch(){ local NAME=$1',
        '  perl -e "alarm 9" get > logs/fetch_$NAME.log 2>&1 \\',
        '    || { say "DL FAIL ($NAME)"; return 11; }',
        '  P) . $W/sc2i_box_p.sh; install_sc2_client ;;',
        '  [ "$BOX" = P ] && prove_p',
        'case "$BOX" in N) box_n;; O) box_o;; P) box_p;; esac', ""])
    lines = base_run.split("\n")
    hold = [lines[0].replace("N|O|P)", "N|O|P|Q)"), lines[1].replace("P)", "P|Q)"),
            lines[2].replace('"huggingface_hub>=0.23"', '"huggingface_hub>=0.23"' + BROTLI), FETCH[0], FETCH[1], lines[3],
            lines[4], FETCH[2], FETCH[3], lines[5], lines[6], Q_LINES[0], lines[7], Q_LINES[1],
            lines[8].replace("P) box_p;;", "P) box_p;; Q) box_q;;"), ""]
    common = {"bench/sc2/sc2f_box_n.sh": "prove_n(){ :; }\n", "bench/sc2/sc2i_box_p.sh": "prove_p(){ :; }\n",
              "bench/sc1/sc1_drive.sh": 'STAGE="$STAGE $REPO/bench/sc2/sc2f_box_n.sh"\n',
              "bench/sc1/make_pin.sh": 'echo "$(sha_of x)  sc2f_box_n.sh"\n'}
    b = _pin({**common, "bench/sc1/sc1_run.sh": base_run})
    newf = {"bench/sc2/" + n: f"# {n}\n" for n in NEW if n.startswith("sc2j")}
    newf.update({"bench/p117/p117_reduce.py": "#\n", "tests/test_decode_graph_buckets.py": "#\n",
                 "bench/kv-pool-g/g3-smoke/g3_smoke.py": "#\n"})
    drv = {"bench/sc1/sc1_drive.sh": common["bench/sc1/sc1_drive.sh"]
           + 'STAGE="$STAGE $REPO/bench/p117/p117_reduce.py $REPO/bench/kv-pool-g/g3-smoke/g3_smoke.py"\n'
           + '    g3_smoke.py) src="$REPO/bench/kv-pool-g/g3-smoke/$name";;\n',
           "bench/sc1/make_pin.sh": common["bench/sc1/make_pin.sh"] + 'echo "$(sha_of g3)  g3_smoke.py"\n'}
    h = _pin({**common, **newf, **drv, "bench/sc1/sc1_run.sh": "\n".join(hold)})
    return b, h


def self_test() -> int:
    global Q_REPLACED
    saved, Q_REPLACED = Q_REPLACED, 3      # the fixture extends three box lists
    try:
        b, h = _fixtures()

        def run(mut_h=None, mut_at=None, mut_b=None):
            bb, hh = dict(b), dict(h)
            if mut_b:
                mut_b(bb)
            if mut_h:
                mut_h(hh)
            if mut_h and PIN in hh and not hh.get("_keep_pin"):
                _pin(hh)
            aa = dict(hh)
            if mut_at:
                mut_at(aa)
            return check(Fake({"B": bb, "H": hh, "L": aa}), "H", base="B", at="L")["verdict"]

        def edit_run(f):
            return lambda t: t.__setitem__("bench/sc1/sc1_run.sh", f(t["bench/sc1/sc1_run.sh"]))
        cases = [
            ("as declared", run(), "OK"),
            ("no hold given", check(Fake({"B": b}), None, base="B")["verdict"], "REFUSE"),
            ("hold unreadable", check(Fake({"B": b}), "H", base="B")["verdict"], "REFUSE"),
            ("pin missing at the hold", run(mut_h=lambda t: t.pop(PIN)), "REFUSE"),
            ("another entry changed", run(mut_h=lambda t: t.__setitem__("bench/sc2/sc2f_box_n.sh", "x\n")), "REFUSE"),
            ("an undeclared entry added", run(mut_h=lambda t: t.__setitem__("bench/sc2/sc2k_box_r.sh", "x\n")),
             "REFUSE"),
            ("a declared new entry absent", run(mut_h=lambda t: t.pop("bench/sc2/sc2j_reduce.py")), "REFUSE"),
            ("an entry removed", run(mut_h=lambda t: t.pop("bench/sc2/sc2i_box_p.sh")), "REFUSE"),
            ("an undeclared run change", run(mut_h=edit_run(lambda s: s.replace("GNF4_SHA=41ef", "GNF4_SHA=0000"))),
             "REFUSE"),
            ("brotli line absent", run(mut_h=edit_run(lambda s: s.replace(BROTLI, ""))), "REFUSE"),
            ("fetch retry absent", run(mut_h=edit_run(lambda s: s.replace(FETCH[2] + "\n", ""))), "REFUSE"),
            ("a box Q line absent", run(mut_h=edit_run(lambda s: s.replace(Q_LINES[1] + "\n", ""))), "REFUSE"),
            ("a non-Q box added", run(mut_h=edit_run(lambda s: s.replace("N|O|P|Q)", "M|N|O|P|Q)"))), "REFUSE"),
            ("an N box file names a new file",
             run(mut_h=lambda t: t.__setitem__("bench/sc2/sc2f_box_n.sh", "python3 $W/sc2j_reduce.py\n")), "REFUSE"),
            ("the driver runs a new file",
             run(mut_h=lambda t: t.__setitem__("bench/sc1/sc1_drive.sh", t["bench/sc1/sc1_drive.sh"]
                                               + "python3 g3_smoke.py\n")), "REFUSE"),
            ("pin does not match its source",
             run(mut_h=lambda t: t.update({"bench/sc2/sc2i_box_p.sh": "y\n", "_keep_pin": True})), "REFUSE"),
            ("launch entry differs", run(mut_at=lambda t: (t.__setitem__("bench/sc2/sc2i_box_p.sh", "z\n"), _pin(t))),
             "REFUSE"),
            ("launch driver differs", run(mut_at=lambda t: t.__setitem__("bench/sc1/sc1_drive.sh", "#\n")), "REFUSE"),
            ("launch pin unreadable", run(mut_at=lambda t: t.pop(PIN)), "REFUSE"),
        ]
    finally:
        Q_REPLACED = saved
    fails = 0
    for name, got, want in cases:
        fails += got != want
        print(f"SC2_KIT_HOLD self-test {'ok ' if got == want else 'BAD'} {name}: {got} (want {want})")
    print(f"SC2_KIT_HOLD self-test {'PASS' if not fails else 'FAIL'} ({len(cases) - fails}/{len(cases)})")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("check",))
    ap.add_argument("--hold", default=HOLD)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--at")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.cmd != "check":
        ap.error("nothing to do")
    rec = check(Git(a.repo), a.hold, base=a.base, at=a.at or "HEAD")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
    print("SC2_KIT_HOLD " + json.dumps({k: rec.get(k) for k in ("verdict", "why", "base", "hold", "at", "changed",
                                                                 "added", "removed", "hold_diff")}), flush=True)
    for x in rec.get("inert_hits") or ():
        print(f"SC2_KIT_HOLD inert {'ok ' if x['allowed'] else 'BAD'} {x['file']}:{x['line_no']}: {x['text'][:120]}")
    return 0 if rec["verdict"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
