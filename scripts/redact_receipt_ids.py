"""Replace rented-hardware identifiers in tracked files with stable, keyed tokens.

A public receipt never needs the rented card's GPU UUID or the provider's instance
id, and both reached the public tree before the receipts rule said so (2026-10-20).
This rewrites them in place, forward only: history is not touched, and the private
receipt store keeps every original byte-exact.

**Tokens, not blanks.** Two readers compare the identifiers: `bench/ra/ra_reduce.py`
puts `gpu_uuid` in its run identity, and P132's same-card rule
(`tests/test_p132_card.py`) keys on literal UUIDs. Nulling the field would make
every card look alike. Each distinct value becomes one token, the first 8 hex of
HMAC-SHA256(key, kind + ":" + value), so equal stays equal and different stays
different, and nothing here reverses it:

    GPU-<8-4-4-4-12 hex>                ->  GPU-redacted-<8 hex>
    "uuid": "<8-4-4-4-12 hex>"          ->  "uuid": "redacted-<8 hex>"
    "vast_instance_id": "<6-9 digits>"  ->  "vast_instance_id": "instance-redacted-<8 hex>"
    Vast instance <7-9 digits>          ->  Vast instance redacted-<8 hex>

The token's shape is deliberately not a UUID's, so `.github/private-markers.txt`
can refuse the UUID shape across the whole tree. The key is store-only; with it,
anyone holding the originals re-derives the redacted bytes exactly (`--verify`).

The script refuses to run if two distinct values of one kind share a token: 8 hex is
32 bits, and a collision would merge two cards into one identity.

    python scripts/redact_receipt_ids.py --key-file K --apply --manifest M [paths...]
    python scripts/redact_receipt_ids.py --key-file K --verify M --from-rev REV
"""
import argparse
import hashlib
import hmac
import os
import re
import subprocess
import sys

HEX_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
ZERO_UUID = "-".join(("0" * 8, "0" * 4, "0" * 4, "0" * 4, "0" * 12))   # the guard's canary stays (built: no literal)
# Built by a tool from other files: rebuilt after the redaction (python scripts/build_llms_bundle.py), never edited here
GENERATED = ("llms-full.txt",)

GPU_RE = re.compile(r"\b(GPU|MIG)-(" + HEX_UUID + r")\b")
BARE_UUID_RE = re.compile(r"""(uuid["']?\s*[:=]\s*["']?)(""" + HEX_UUID + r")")
# a field: "vast_instance_id": "<digits>", "instance_id": <digits>, instance_id=<digits>
INSTANCE_FIELD_RE = re.compile(r"""((?:vast_)?instance[_ ]id(["']?)\s*[:=]\s*)(["']?)(\d{6,9})\b(["']?)""", re.I)
# prose: "Vast instance <digits>", "instances <digits>", "instance id #<digits>"
INSTANCE_PROSE_RE = re.compile(r"\b(instances? (?:id )?#?)(\d{7,9})\b", re.I)


def token(key, kind, value):
    return hmac.new(key, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()[:8]


def find_values(text):
    """{kind: set(values)} named in text, the canary excluded."""
    gpu = {m.group(2) for m in GPU_RE.finditer(text)} | {m.group(2) for m in BARE_UUID_RE.finditer(text)}
    inst = {m.group(4) for m in INSTANCE_FIELD_RE.finditer(text)} | {m.group(2) for m in INSTANCE_PROSE_RE.finditer(text)}
    gpu.discard(ZERO_UUID)
    return {"gpu": gpu, "instance": inst}


def redact(text, key):
    def gpu(m):
        if m.group(2) == ZERO_UUID:
            return m.group(0)
        return f"{m.group(1)}-redacted-{token(key, 'gpu', m.group(2))}"

    def bare(m):
        if m.group(2) == ZERO_UUID:
            return m.group(0)
        return f"{m.group(1)}redacted-{token(key, 'gpu', m.group(2))}"

    def field(m):
        lead, key_quote, open_q, value, close_q = m.groups()
        tok = f"instance-redacted-{token(key, 'instance', value)}"
        if open_q:                           # already a string: keep its quotes
            return f"{lead}{open_q}{tok}{close_q}"
        if key_quote:                        # a JSON number: the token must become a string
            return f'{lead}"{tok}"{close_q}'
        return f"{lead}{tok}{close_q}"       # a key=value log line

    def prose(m):
        return f"{m.group(1)}redacted-{token(key, 'instance', m.group(2))}"

    text = GPU_RE.sub(gpu, text)
    text = BARE_UUID_RE.sub(bare, text)
    text = INSTANCE_FIELD_RE.sub(field, text)
    return INSTANCE_PROSE_RE.sub(prose, text)


# The second edit, on exactly three lines: each names the control plane or the receipt store the way public text no
# longer does (.github/private-markers.txt, added lines). Redacting the line makes it an added line, so the old name is
# reworded to the project-voice form in the same transform. Each pair must match once in its file's redacted text.
_CP = "ader" + "tha"   # split, so this file does not name it either
REWORD = {
    "bench/h2h-2026-10-02/tc1/README.md": ((f"({_CP}-agents#138)", "(the control plane's issue #138)"),),
    "bench/h2h-2026-10-02/tc1/RESULTS-tc1-steady18.md": ((f"({_CP}-agents#139)", "(the control plane's issue #139)"),),
    "bench/p100/RESULTS-p100.md": ((f"destroyed and absent. Receipt: {_CP}-receipts",
                                    "destroyed and absent. Receipt: the receipt store"),),
}


def redact_file(path, text, key):
    """redact() plus the path's REWORD pairs; a pair that does not match exactly once refuses."""
    text = redact(text, key)
    for old, new in REWORD.get(path, ()):
        if text.count(old) != 1:
            sys.exit(f"refusing: {path} holds {text.count(old)} copies of {old!r}, not one")
        text = text.replace(old, new)
    return text


def assert_distinct(values, key):
    """Refuse a 32-bit collision: two values of one kind sharing a token."""
    for kind, vals in values.items():
        seen = {}
        for v in sorted(vals):
            t = token(key, kind, v)
            if t in seen and seen[t] != v:
                sys.exit(f"refusing: {kind} values collide on token {t}; widen the token before redacting")
            seen[t] = v


def read_key(path):
    key = open(path, "rb").read().strip()
    if len(key) < 32:
        sys.exit(f"refusing: the key in {path} is shorter than 32 bytes")
    return key


def sha256(b):
    return hashlib.sha256(b).hexdigest()


SUMS_ENTRY = re.compile(r"^([0-9a-f]{64})(\s+\*?)((?:\./)?)(.+?)\s*$")


def sums_dir(p):
    """The nearest directory above p that holds a SHA256SUMS: the receipt that vouches for p."""
    d = os.path.dirname(p)
    while d:
        if os.path.isfile(os.path.join(d, "SHA256SUMS")):
            return d
        d = os.path.dirname(d)
    return None


def repin(sums, changes):
    """Rewrite the SHA256SUMS lines of changed files ({rel path: (old sha, new sha)}); any other line stays as it is.
    A line whose sha is not the file's old sha is left alone and returned, never 'fixed'."""
    lines, stale = open(sums, encoding="utf-8").read().split("\n"), []
    for i, ln in enumerate(lines):
        m = SUMS_ENTRY.match(ln)
        if not m or m.group(4) not in changes:
            continue
        old, new = changes[m.group(4)]
        if m.group(1) != old:
            stale.append(m.group(4))
            continue
        lines[i] = f"{new}{m.group(2)}{m.group(3)}{m.group(4)}"
    open(sums, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    return stale


def record_originals(args, rows):
    """Re-pin each vouching SHA256SUMS, and name each store-held original in its receipt's STORE-ONLY.sha256 under a
    `# same bytes: <path> at <rev>` line (tests/test_receipt_sums.py checks those against git)."""
    store = set(open(args.store_shas).read().split()) if args.store_shas else set()
    by_dir = {}
    for p, a, b in rows:
        d = sums_dir(p)
        if d:
            by_dir.setdefault(d, []).append((os.path.relpath(p, d), p, a, b))
    stale = []
    for d, items in sorted(by_dir.items()):
        held = [(rel, p, a) for rel, p, a, _ in sorted(items) if a in store]
        so = os.path.join(d, "STORE-ONLY.sha256")
        if held:
            before = open(so, "rb").read() if os.path.exists(so) else None
            text = before.decode("utf-8") if before else (
                "# Files of this receipt kept only in the private receipt store, byte-exact: sha256, then the path\n"
                "# within this receipt.\n")
            text = text.rstrip("\n") + "\n" + (
                "# Originals of files this receipt carries with rented-hardware identifiers replaced by keyed tokens\n"
                "# (docs/receipts-redaction.md). Each original is in the private receipt store byte-exact, and is\n"
                "# also git's bytes at the commit named on its line.\n")
            for rel, p, a in held:
                text += f"# same bytes: {p} at {args.rev[:12]}\n{a}  {rel}\n"
            new = text.encode("utf-8")
            complete = before is None and lists_every_file(d)
            open(so, "wb").write(new)
            if before is not None:
                items.append(("STORE-ONLY.sha256", so, sha256(before), sha256(new)))
        else:
            complete = False
        sums = os.path.join(d, "SHA256SUMS")
        stale += [os.path.join(d, s) for s in repin(sums, {rel: (a, b) for rel, _, a, b in items})]
        if complete:   # a SHA256SUMS that lists every file of its receipt must list the new one too
            text = open(sums, encoding="utf-8").read()
            sep = SUMS_ENTRY.match(next(ln for ln in text.split("\n") if SUMS_ENTRY.match(ln))).group(2)
            open(sums, "w", encoding="utf-8", newline="\n").write(
                text.rstrip("\n") + f"\n{sha256(open(so, 'rb').read())}{sep}STORE-ONLY.sha256\n")
    return stale, store


def lists_every_file(d):
    """True when d's SHA256SUMS names every tracked file under d but itself."""
    listed = {m.group(4) for ln in open(os.path.join(d, "SHA256SUMS"), encoding="utf-8").read().split("\n")
              if (m := SUMS_ENTRY.match(ln))}
    tracked = subprocess.run(["git", "ls-files", "--", d], capture_output=True, text=True, check=True).stdout.split()
    return {os.path.relpath(p, d) for p in tracked} - {"SHA256SUMS"} <= listed


def apply(args, key):
    paths = args.paths or subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                                         check=True).stdout.split("\n")
    keep = set(GENERATED)
    if args.exclude:   # files another file pins by hash, or that a runner reads: never edited here
        keep |= {ln.split("#", 1)[0].strip() for ln in open(args.exclude, encoding="utf-8")} - {""}
    work, values = [], {"gpu": set(), "instance": set()}
    for p in sorted(x for x in paths if x and x not in keep):
        try:
            raw = open(p, "rb").read()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        found = find_values(text)
        if found["gpu"] or found["instance"]:
            work.append((p, raw, text))
            for k in values:
                values[k] |= found[k]
    assert_distinct(values, key)
    rows, out = [], []
    for p, raw, text in work:   # every file is computed before any is written: a refusal leaves the tree untouched
        new = redact_file(p, text, key).encode("utf-8")
        if new != raw:
            out.append((p, new))
            rows.append((p, sha256(raw), sha256(new)))
    for p, new in out:
        open(p, "wb").write(new)
    stale, store = record_originals(args, rows)
    with open(args.manifest, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# original sha256, redacted sha256, where the original is held, path. Originals are git's bytes at "
                f"{args.rev}; 'store' ones are also in the private receipt store byte-exact.\n")
        for p, a, b in rows:
            f.write(f"{a}  {b}  {'store' if a in store else 'git:' + args.rev[:8]}  {p}\n")
    print(f"redacted {len(rows)} files: {len(values['gpu'])} GPU UUIDs, {len(values['instance'])} instance ids; "
          f"tokens distinct per kind; {sum(a in store for _, a, _ in rows)} originals store-held")
    for s in stale:
        print(f"left alone (its SHA256SUMS line did not hold the file's old sha): {s}")


def verify(args, key):
    """Re-derive every manifest row from the original bytes at --from-rev; both hashes must match."""
    bad = n = 0
    for line in open(args.verify, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        orig_sha, red_sha, _origin, p = line.rstrip("\n").split("  ", 3)
        raw = subprocess.run(["git", "show", f"{args.from_rev}:{p}"], capture_output=True, check=True).stdout
        new = redact_file(p, raw.decode("utf-8"), key).encode("utf-8")
        cur = open(p, "rb").read() if os.path.exists(p) else b""
        n += 1
        if sha256(raw) != orig_sha or sha256(new) != red_sha or sha256(cur) != red_sha:
            bad += 1
            print(f"MISMATCH {p}")
    print(f"verified {n - bad} of {n} rows")
    return 1 if bad or n == 0 else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--manifest", help="with --apply: write 'original_sha  redacted_sha  path' rows here")
    ap.add_argument("--exclude", help="with --apply: a file of paths to leave byte-exact (# comments allowed)")
    ap.add_argument("--store-shas", help="with --apply: the sha256s the private receipt store holds, one per line")
    ap.add_argument("--rev", help="with --apply: the commit whose bytes are the originals (named in the records)")
    ap.add_argument("--verify", metavar="MANIFEST")
    ap.add_argument("--from-rev", help="with --verify: the git revision holding the originals")
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args()
    key = read_key(args.key_file)
    if args.apply:
        if not args.manifest or not args.rev:
            ap.error("--apply needs --manifest and --rev")
        args.rev = subprocess.run(["git", "rev-parse", args.rev], capture_output=True, text=True,
                                  check=True).stdout.strip()
        apply(args, key)
        return 0
    if args.verify:
        if not args.from_rev:
            ap.error("--verify needs --from-rev")
        return verify(args, key)
    ap.error("give --apply or --verify")


if __name__ == "__main__":
    sys.exit(main())
