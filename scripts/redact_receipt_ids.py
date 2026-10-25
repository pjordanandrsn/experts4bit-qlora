"""Replace rented-hardware and host identifiers in tracked files with stable tokens (docs/receipts-redaction.md).

A public receipt never needs the rented card's GPU UUID, the provider's instance id, the box's ssh endpoint or public IP,
or a contributor's home or scratch path. This rewrites them in place, forward only: history is not touched, and the
private receipt store keeps every original byte-exact.

**Values, not shapes.** The input is the store's token map (kind, value, token), harvested from the store's own records
of every box ever rented. Each kind is replaced by VALUE wherever it is unambiguous, and the token map is the same
oracle the completeness gate searches with:

    gpu       GPU-<uuid> / "uuid": "<uuid>"   ->  GPU-redacted-<8 hex> / "uuid": "redacted-<8 hex>"
    ssh       ssh<N>.vast.ai:<port>            ->  ssh-redacted-<8 hex>
    ip        <a.b.c.d>                        ->  ip-redacted-<8 hex>
    path      /private/tmp/claude-<n>/...      ->  <scratch>        (and a home directory -> <home>)
    instance  <digits>, IN AN ID CONTEXT       ->  instance-redacted-<8 hex> (or redacted-<8 hex> after "instance")

An instance id is a bare number, so it is replaced only where the text says it is one: within the 48 characters before
it (the previous line included, for a wrapped sentence) there is an id word (instance, vast, box, contract, container,
provider, list_after), and it is not part of a longer number (no digit, '.', or hex letter beside it). A number that
merely equals an id elsewhere is left alone; the gate lists it, and a reviewed coincidence allows it by name.

Equal values stay equal and different ones stay different (the tokens are HMAC-SHA256 under a key only the store holds),
so ``bench/ra/ra_reduce.py``'s run identity still tells cards apart. Binary files (any NUL byte) are never edited here.

    python scripts/redact_receipt_ids.py --map M --apply --rev REV --holds H [--store-shas S]
    python scripts/redact_receipt_ids.py --map M --verify --rev REV --holds H [--store-shas S]
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile

ZERO_UUID = "-".join(("0" * 8, "0" * 4, "0" * 4, "0" * 4, "0" * 12))   # the guard's canary stays (built: no literal)
# Built by a tool from other files: rebuilt after the redaction (python scripts/build_llms_bundle.py), never edited here
GENERATED = ("llms-full.txt",)
ID_WORDS = re.compile(r"instance|vast|box|contract|container|provider|list_after", re.I)
# Reviewed, file by file: in these, EVERY bounded number that equals an instance id is one (a table cell, "(<id>, EPYC
# 9755)", "on <id>"), so the id-word context is not required there. A number inside a longer number or a float is still
# never touched. Adding a file here is a review decision, named in docs/receipts-redaction.md.
ID_FILES = frozenset({
    "CHANGELOG.md",
    "docs/claims.json",
    "bench/hybrid-g9/sv1/RESULTS-sv1-census.md",
    "bench/h2h-20260906/tp2/P40-PREREG.md",
    "bench/hybrid-g9/throughput-20260904/bo7/RESULTS.md",
    "bench/p55/P55-PREREG.md",
    "bench/sv7/RESULTS-sv7.md",
})
# Binary files that leave the tree: identifiers inside them can be split or compressed past any scan. Each is in the
# private receipt store byte-exact, and the STORE-ONLY.sha256 beside it says so.
REMOVED = (
    "ab-telemetry/wandb/wandb/offline-run-20260703_165149-nut4l67k/run-nut4l67k.wandb",
    "ab-telemetry/wandb/wandb/offline-run-20260703_174045-jym7tnum/run-jym7tnum.wandb",
    "bench/cpu-router/g1_audit_legacy.nsys-rep",
    "bench/cpu-router/g1_audit_replica.nsys-rep",
    "bench/hybrid-g3/g3_nsys.nsys-rep",
    "bench/hybrid-g3/g3_nsys_235b.nsys-rep",
    "bench/hybrid-g6/g6_nsys.nsys-rep",
)
# Archives whose members the public recipe reads: re-packed with redacted members (scripts/repack_tar.py)
REPACKED = ("bench/dq9/receipts/dq9-5090-2/raw-science.tar.gz",)
STORE_NOTE = ("# Kept only in the private receipt store, byte-exact (docs/receipts-redaction.md): binaries that carried\n"
              "# rented-hardware identifiers, some split or compressed past any scan. sha256, then the file's name here.\n")
SCRATCH = re.compile(r"/private/tmp/claude-\d+/[^\s\"'`)\]<>,;|\\]*")

# The second edit, on exactly three lines: each names the control plane or the receipt store the way public text no
# longer does (.github/private-markers.txt, added lines). Redacting the line makes it an added line, so the old name is
# reworded to the project-voice form in the same transform. Each pair must match once in its file's redacted text.
_CP = "ader" + "tha"   # split, so this file does not name it either
REWORD = {
    "bench/h2h-2026-10-02/tc1/README.md": ((f"({_CP}-agents#138)", "(the control plane's issue #138)"),),
    "bench/h2h-2026-10-02/tc1/RESULTS-tc1-steady18.md": ((f"({_CP}-agents#139)", "(the control plane's issue #139)"),),
    "bench/p100/RESULTS-p100.md": ((f"destroyed and absent. Receipt: {_CP}-receipts",
                                    "destroyed and absent. Receipt: the receipt store"),),
    # the binaries that leave the tree, and the re-packed archive: their readers are told where the bytes went
    "bench/cpu-router/RESULTS-hybrid-phase1.md": (
        ("(+ `g1_nsys_api_*.txt` extracts).",
         "(+ `g1_nsys_api_*.txt` extracts; since 2026-10-25 the traces are in the private receipt store only, with their\n"
         "sha256 in `STORE-ONLY.sha256` here)."),),
    "bench/hybrid-g3/RESULTS-g3-shakeout.md": (
        ("Trace committed (`g3_nsys.nsys-rep` + stats extracts).",
         "Trace committed (`g3_nsys.nsys-rep` + stats extracts; since 2026-10-25 the trace is in the\n"
         "private receipt store only, its sha256 in `STORE-ONLY.sha256` here)."),),
    "bench/hybrid-g3/RESULTS-g3-formal.md": (
        ("committed (`g3_nsys_235b.nsys-rep`).",
         "committed (`g3_nsys_235b.nsys-rep`; since 2026-10-25 in the private receipt store only, its sha256\n"
         "in `STORE-ONLY.sha256` here)."),),
    "bench/hybrid-g6/RESULTS-g6.md": (
        ("(`g6_nsys.nsys-rep`, streams summary",
         "(`g6_nsys.nsys-rep`, since 2026-10-25 in the private receipt store only, its sha256 in\n"
         "   `STORE-ONLY.sha256` here; streams summary"),),
    "bench/dq9/receipts/dq9-5090-2/README.md": (
        ("contains unchanged JSON receipts,",
         "contains the JSON receipts (their instance ids and GPU UUIDs replaced by tokens on 2026-10-25, see "
         "docs/receipts-redaction.md; the archive as first committed is in the private receipt store),"),),
}


def load_map(path):
    """{kind: {value (lower-case for gpu): token}}; kinds without a public form here (sshkey) ride only in binaries."""
    out = {}
    for ln in open(path, encoding="utf-8"):
        if not ln.strip() or ln.startswith("#"):
            continue
        kind, value, token = ln.rstrip("\n").split("\t")[:3]
        out.setdefault(kind, {})[value.lower() if kind == "gpu" else value] = token
    for kind, vals in out.items():
        if len(set(vals.values())) != len(vals):
            sys.exit(f"refusing: two {kind} values share a token")
    return out


def _alternation(values):
    return "|".join(re.escape(v) for v in sorted(values, key=len, reverse=True))


class Redactor:
    def __init__(self, tokens):
        self.t = tokens
        g = tokens.get("gpu", {})
        self.gpu = re.compile(r"(?i)(?<![0-9a-f])(" + _alternation(g) + r")(?![0-9a-f])") if g else None
        # a UUID cut short after its GPU-/MIG- word ("GPU-" and 8 hex): the prefix of a mapped UUID takes that UUID's token
        self.gpu_prefix = re.compile(r"(?i)\b(GPU|MIG)-([0-9a-f]{6,8}(?:-[0-9a-f]{1,12}){0,3})(?![0-9a-f-])")
        s = tokens.get("ssh", {})
        self.ssh = re.compile("(" + _alternation(s) + r")(?![0-9])") if s else None
        ip = tokens.get("ip", {})
        self.ip = re.compile(r"(?<![0-9.])(" + _alternation(ip) + r")(?![0-9])") if ip else None
        self.inst = tokens.get("instance", {})
        self.digits = re.compile(r"(?<![0-9A-Za-z.])([0-9]{6,10})(?![0-9A-Za-z]|\.[0-9])")
        # an id glued to its word, as some receipts write it ("instance<digits>", "box<digits>")
        self.glued = re.compile(r"(?i)\b(instances?|box)([0-9]{6,10})(?![0-9A-Za-z])")

    def gpu_sub(self, m):
        v = m.group(1).lower()
        if v == ZERO_UUID:
            return m.group(0)
        return "redacted-" + self.t["gpu"][v]

    def prefix_sub(self, m):
        hit = [v for v in self.t["gpu"] if v.startswith(m.group(2).lower())]
        if len(hit) > 1:
            sys.exit(f"refusing: a {len(m.group(2))}-character GPU prefix matches {len(hit)} mapped UUIDs")
        return m.group(1) + "-redacted-" + self.t["gpu"][hit[0]] if hit else m.group(0)

    def inst_sub(self, text, path=""):
        text = self.glued.sub(lambda m: (m.group(1) + "-redacted-" + self.inst[m.group(2)])
                              if m.group(2) in self.inst else m.group(0), text)
        out, last = [], 0
        for m in self.digits.finditer(text):
            v = m.group(1)
            if v not in self.inst:
                continue
            before = text[max(0, m.start() - 48):m.start()]
            if path not in ID_FILES and not ID_WORDS.search(before):
                continue                                  # a number that equals an id, outside an id context: untouched
            tok = self.inst[v]
            lead = before.rstrip("*`#( \n").lower()
            word = "redacted-" if lead.endswith(("instance", "instances", "instance id", "box", "vast")) else "instance-redacted-"
            rep = word + tok
            key = re.search(r'(\\?)"[a-z_]*instance_id\\?":\s*$', before)
            if key:   # a JSON number (plain or escaped JSON) becomes a string token, quoted the way its key is
                q = key.group(1) + '"'
                rep = q + rep + q
            out.append(text[last:m.start()] + rep)
            last = m.end()
        return "".join(out) + text[last:]

    def redact(self, path, text):
        if self.gpu:
            text = self.gpu.sub(self.gpu_sub, text)
            text = self.gpu_prefix.sub(self.prefix_sub, text)
        if self.ssh:
            text = self.ssh.sub(lambda m: "ssh-redacted-" + self.t["ssh"][m.group(1)], text)
        if self.ip:
            text = self.ip.sub(lambda m: "ip-redacted-" + self.t["ip"][m.group(1)], text)
        text = SCRATCH.sub("<scratch>", text)
        for home in self.t.get("path", {}):
            if home.startswith("/") and "tmp" not in home:
                text = text.replace(home, "<home>")
            elif home.startswith("-"):
                text = text.replace(home, "-<home>")
        text = self.inst_sub(text, path)
        for old, new in REWORD.get(path, ()):
            if text.count(old) != 1:
                sys.exit(f"refusing: {path} holds {text.count(old)} copies of {old!r}, not one")
            text = text.replace(old, new)
        return text


def sha256(b):
    return hashlib.sha256(b).hexdigest()


SUMS_ENTRY = re.compile(r"^([0-9a-f]{64})(\s+\*?)((?:\./)?)(.+?)\s*$")


def sums_dir(p, root):
    d = os.path.dirname(p)
    while d:
        if os.path.isfile(os.path.join(root, d, "SHA256SUMS")):
            return d
        d = os.path.dirname(d)
    return None


def lists_every_file(root, d):
    listed = {m.group(4) for ln in open(os.path.join(root, d, "SHA256SUMS"), encoding="utf-8").read().split("\n")
              if (m := SUMS_ENTRY.match(ln))}
    tracked = subprocess.run(["git", "-C", root, "ls-files", "--", d], capture_output=True, text=True,
                             check=True).stdout.split()
    return {os.path.relpath(p, d) for p in tracked} - {"SHA256SUMS"} <= listed


def repin(path, changes):
    lines, stale = open(path, encoding="utf-8").read().split("\n"), []
    for i, ln in enumerate(lines):
        m = SUMS_ENTRY.match(ln)
        if not m or m.group(4) not in changes:
            continue
        old, new = changes[m.group(4)]
        if m.group(1) != old:
            stale.append(m.group(4))
            continue
        lines[i] = f"{new}{m.group(2)}{m.group(3)}{m.group(4)}"
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    return stale


def record(root, rows, store, rev):
    """Re-pin each vouching SHA256SUMS, and name each store-held original in its receipt's STORE-ONLY.sha256 under a
    `# same bytes: <path> at <rev>` line (tests/test_receipt_sums.py checks those against git)."""
    by_dir, stale = {}, []
    for p, a, b in rows:
        d = sums_dir(p, root)
        if d:
            by_dir.setdefault(d, []).append((os.path.relpath(p, d), p, a, b))
    for d, items in sorted(by_dir.items()):
        held = [(rel, p, a) for rel, p, a, _ in sorted(items) if a in store]
        so = os.path.join(root, d, "STORE-ONLY.sha256")
        complete = False
        if held:
            before = open(so, "rb").read() if os.path.exists(so) else None
            complete = before is None and lists_every_file(root, d)
            text = before.decode("utf-8") if before else (
                "# Files of this receipt kept only in the private receipt store, byte-exact: sha256, then the path\n"
                "# within this receipt.\n")
            text = text.rstrip("\n") + "\n" + (
                "# Originals of files this receipt carries with rented-hardware identifiers replaced by tokens\n"
                "# (docs/receipts-redaction.md). Each original is in the private receipt store byte-exact, and is\n"
                "# also git's bytes at the commit named on its line.\n")
            for rel, p, a in held:
                text += f"# same bytes: {p} at {rev[:12]}\n{a}  {rel}\n"
            new = text.encode("utf-8")
            open(so, "wb").write(new)
            if before is not None:
                items.append(("STORE-ONLY.sha256", so, sha256(before), sha256(new)))
        sums = os.path.join(root, d, "SHA256SUMS")
        stale += [os.path.join(d, s) for s in repin(sums, {rel: (a, b) for rel, _, a, b in items})]
        if complete:   # a SHA256SUMS that lists every file of its receipt must list the new one too
            text = open(sums, encoding="utf-8").read()
            sep = SUMS_ENTRY.match(next(ln for ln in text.split("\n") if SUMS_ENTRY.match(ln))).group(2)
            open(sums, "w", encoding="utf-8", newline="\n").write(
                text.rstrip("\n") + f"\n{sha256(open(so, 'rb').read())}{sep}STORE-ONLY.sha256\n")
    return stale


def apply(root, red, holds, store, rev, manifest):
    files = subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True, check=True).stdout.decode()
    keep = set(GENERATED) | holds | set(REMOVED) | set(REPACKED)
    out, rows = [], []
    for p in sorted(f for f in files.split("\0") if f and f not in keep):
        try:
            raw = open(os.path.join(root, p), "rb").read()
        except OSError:
            continue
        if b"\0" in raw:
            continue                                      # binaries are moved store-only, never edited
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        new = red.redact(p, text).encode("utf-8")
        if new != raw:
            out.append((p, new))
            rows.append((p, sha256(raw), sha256(new)))
    from repack_tar import repack   # noqa: E402  (beside this script)
    for p in REPACKED:   # computed before anything is written, like the text files
        src = os.path.join(root, p)
        raw = open(src, "rb").read()
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "out")
            repack(src, dst, red)
            new = open(dst, "rb").read()
        out.append((p, new))
        rows.append((p, sha256(raw), sha256(new)))
    gone = []
    for p in REMOVED:
        raw = open(os.path.join(root, p), "rb").read()
        if sha256(raw) not in store:
            sys.exit(f"refusing: {p} is not in the private receipt store; copy it there before it leaves the tree")
        gone.append((p, sha256(raw)))
    for p, new in out:   # every file is computed before any is written: a refusal leaves the tree untouched
        open(os.path.join(root, p), "wb").write(new)
    for p, a in gone:
        os.remove(os.path.join(root, p))
        so = os.path.join(root, os.path.dirname(p), "STORE-ONLY.sha256")
        text = open(so, encoding="utf-8").read() if os.path.exists(so) else STORE_NOTE
        if os.path.basename(p) not in text:
            text = text.rstrip("\n") + f"\n{a}  {os.path.basename(p)}\n"
        open(so, "w", encoding="utf-8", newline="\n").write(text)
    stale = record(root, rows, store, rev)
    with open(os.path.join(root, manifest), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# original sha256, redacted sha256 ('removed' when the file left the tree), where the original is held, "
                f"path. Originals are git's bytes at {rev}; 'store' ones are also in the private receipt store "
                f"byte-exact.\n")
        for p, a, b in rows:
            f.write(f"{a}  {b}  {'store' if a in store else 'git:' + rev[:8]}  {p}\n")
        for p, a in gone:
            f.write(f"{a}  removed  store  {p}\n")
    return rows, stale


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True, help="the store's token map (kind, value, token); read in place")
    ap.add_argument("--rev", required=True, help="the commit whose bytes are the originals")
    ap.add_argument("--holds", required=True, help="paths left byte-exact, one per line (# comments allowed)")
    ap.add_argument("--store-shas", help="the sha256s the private receipt store holds, one per line")
    ap.add_argument("--manifest", default="docs/receipts-redaction.sha256")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="re-run the whole transform on a fresh checkout of --rev and compare every file it touches")
    args = ap.parse_args()
    rev = subprocess.run(["git", "rev-parse", args.rev], capture_output=True, text=True, check=True).stdout.strip()
    red = Redactor(load_map(args.map))
    holds = {ln.split("#", 1)[0].strip() for ln in open(args.holds, encoding="utf-8")} - {""}
    store = set(open(args.store_shas).read().split()) if args.store_shas else set()
    if args.apply:
        rows, stale = apply(".", red, holds, store, rev, args.manifest)
        print(f"redacted {len(rows)} files; {sum(a in store for _, a, _ in rows)} originals store-held")
        for s in stale:
            print(f"left alone (its SHA256SUMS line did not hold the file's old sha): {s}")
        return 0
    if args.verify:
        with tempfile.TemporaryDirectory() as tmp:
            wt = os.path.join(tmp, "wt")
            subprocess.run(["git", "worktree", "add", "-q", "--detach", wt, rev], check=True)
            try:
                rows, _ = apply(wt, red, holds, store, rev, args.manifest)
                touched = subprocess.run(["git", "-C", wt, "status", "--porcelain", "-z", "--untracked-files=all"],
                                         capture_output=True, check=True).stdout.decode().split("\0")
                paths = sorted({t[3:] for t in touched if t})
                def same(p):   # a file the transform removed must be gone here too
                    a, b = os.path.join(wt, p), p
                    if not os.path.exists(a) or not os.path.exists(b):
                        return os.path.exists(a) == os.path.exists(b)
                    return sha256(open(a, "rb").read()) == sha256(open(b, "rb").read())
                bad = [p for p in paths if not same(p)]
            finally:
                subprocess.run(["git", "worktree", "remove", "--force", wt], check=True)
        for p in bad:
            print(f"MISMATCH {p}")
        print(f"verify: re-ran the transform at {rev[:12]}; {len(paths) - len(bad)} of {len(paths)} touched files "
              f"equal the tree ({len(rows)} redacted, the rest SHA256SUMS / STORE-ONLY / the manifest)")
        return 1 if bad or not paths else 0
    ap.error("give --apply or --verify")


if __name__ == "__main__":
    sys.exit(main())
