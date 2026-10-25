"""The byte guard for rented-hardware identifiers (docs/receipts-redaction.md): run by private-marker-guard.

`git grep -I` skips any file git calls binary, and a profiler dump or a gzip member is exactly where a GPU UUID or an
instance id hides. So this reads BYTES:

- **Whole tree** (`--tree`): every tracked file's raw bytes, and each `.gz` file's decompressed bytes, are refused if
  they hold a GPU UUID (`GPU-<uuid>`, or a bare UUID after a `uuid` key). The guard's canary and the pinned redaction
  holds (`#exempt-fixture:` lines in .github/private-markers.txt, each held to its sha256) are skipped.
- **Added lines** (`--added --base REV`): every line a change adds, binaries included, is refused if it names an
  instance id in any form the redaction met (a JSON key, escaped JSON, `list_after`, "Vast instance <n>", "box <n>",
  a container label, an id glued to its word, or "instance" at a line's end with the id opening the next), a GPU UUID
  or any GPU-/MIG- prefix of one (6 or more hex), a provider ssh endpoint, a public IP in an environment dump, after a `public_ipaddr` key or after an `@`, or a home
  directory.

Old identifier-free numbers never trip it: instance ids are judged by their context, in added lines only.

    python scripts/check_public_ids.py --tree
    python scripts/check_public_ids.py --added --base origin/main
"""
import argparse
import gzip
import hashlib
import ipaddress
import re
import subprocess
import sys

MARKERS = ".github/private-markers.txt"
GUARD_FILES = {MARKERS, ".github/workflows/private-marker-guard.yml", "scripts/check_public_ids.py"}
HEX = rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
UUID = re.compile(rb"(?i)(?:GPU|MIG)-(" + HEX + rb")|uuid[\"']?\s*[:=]\s*[\"']?(" + HEX + rb")")
ZERO = b"-".join((b"0" * 8, b"0" * 4, b"0" * 4, b"0" * 4, b"0" * 12))
ID = rb"[0-9]{6,10}(?![0-9])"
ADDED = [
    ("instance id after its key", re.compile(rb"(?i)(?:vast_|provider_)?instance[_ ]id\\?[\"']?\s*[:=]\s*\\?[\"']?" + ID)),
    ("instance id in list_after", re.compile(rb"(?i)list_after\\?[\"']?\s*:\s*\[\s*\\?[\"']?" + ID)),
    ("instance id in prose", re.compile(rb"(?i)(?<![0-9a-z_])(?:instances?|vast(?: box| instance)?|box)\s+(?:id\s+)?[#*`(]*" + ID)),
    ("instance id glued to its word", re.compile(rb"(?i)(?<![0-9a-z_])(?:instances?|box)" + ID)),
    ("container label", re.compile(rb"(?i)CONTAINERLABEL=C\.[0-9]{6,10}|(?<![0-9A-Za-z])C\.[0-9]{7,10}(?![0-9])")),
    ("provider ssh endpoint",
     re.compile(rb"(?i)ssh[0-9]+\.vast\.ai[: ]+(?:-p\s*)?[0-9]{2,5}|-p\s*[0-9]{2,5}\s+\S*@ssh[0-9]+\.vast\.ai")),
    ("public IP in an environment dump", re.compile(rb"PUBLIC_IPADDR=(?:[0-9]{1,3}\.){3}[0-9]{1,3}")),
    ("GPU UUID or its prefix", re.compile(rb"(?i)(?<![0-9a-z])(?:GPU|MIG)-(?!0{8}-0{4}-0{4}-0{4}-0{12})[0-9a-f]{6}")),
    ("home directory", re.compile(rb"/Users/[a-z][a-z0-9_.-]+/|/home/[a-z][a-z0-9_.-]+/|/private/tmp/claude-[0-9]+/")),
]
IPV4 = rb"((?:[0-9]{1,3}\.){3}[0-9]{1,3})(?![0-9.-])"
# a public address after its key or an "@": global only, so loopback, private, CGNAT and documentation ranges pass
ADDRESSED = [
    ("public IP after an address key", re.compile(rb"(?i)public_ipaddr\\?[\"']?\s*[:=]\s*\\?[\"']?" + IPV4)),
    ("public IP after an @", re.compile(rb"@" + IPV4)),
]


def is_global(ip):
    try:
        a = ipaddress.ip_address(ip.decode())
    except ValueError:
        return False
    return a.is_global and a not in ipaddress.ip_network("100.64.0.0/10")


WRAP_END = re.compile(rb"(?i)(?<![0-9a-z_])(?:instances?|vast(?: box| instance)?|box)\s*$")
WRAP_START = re.compile(rb"^\s*[#*`(]*" + ID)
ALLOW = b"guard-allow"


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def exempt():
    """{path: (cap, sha256)} from the #exempt-fixture lines; a fixture stands only while it holds its pin."""
    out = {}
    for ln in open(MARKERS, encoding="utf-8"):
        if ln.startswith("#exempt-fixture: "):
            _, path, cap, pin = ln.split()
            out[path] = (int(cap), pin)
    return out


def held(path, b, ex):
    cap, pin = ex[path]
    return b.count(b"\n") + (not b.endswith(b"\n")) <= cap and hashlib.sha256(b).hexdigest() == pin


def tree():
    ex, bad = exempt(), []
    for path in git("ls-files", "-z").decode().split("\0"):
        if not path or path in GUARD_FILES:
            continue
        try:
            b = open(path, "rb").read()
        except OSError:
            continue
        if path in ex:
            if not held(path, b, ex):
                bad.append(f"{path}: an exempt fixture that no longer holds its pin")
            continue
        blobs = [b]
        if path.endswith(".gz"):
            try:
                blobs.append(gzip.decompress(b))
            except OSError:
                pass
        for x in blobs:
            for m in UUID.finditer(x):
                if (m.group(1) or m.group(2)).lower() != ZERO:
                    bad.append(f"{path}: a GPU UUID at byte {m.start()}")
                    break
    return bad


def added(base):
    mb = git("merge-base", base, "HEAD").decode().strip()
    diff = git("diff", "-U0", "--no-color", "--text", mb, "HEAD", "--", ".",
               *[f":!{p}" for p in sorted(GUARD_FILES)])
    bad, path, prev = [], "?", b""
    for ln in diff.split(b"\n"):
        if ln.startswith(b"+++ "):
            path, prev = ln[6:].decode(errors="replace") if ln.startswith(b"+++ b/") else ln[4:].decode(), b""
            continue
        if not ln.startswith(b"+") or ALLOW in ln:
            prev = b""
            continue
        line = ln[1:]
        for why, rx in ADDED:
            if rx.search(line):
                bad.append(f"{path}: {why}")
                break
        else:
            hit = next((why for why, rx in ADDRESSED for m in rx.finditer(line) if is_global(m.group(1))), None)
            if hit:
                bad.append(f"{path}: {hit}")
            elif prev and WRAP_END.search(prev) and WRAP_START.search(line):
                bad.append(f"{path}: an instance id wrapped onto the next line")
        prev = line
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", action="store_true")
    ap.add_argument("--added", action="store_true")
    ap.add_argument("--base", default="origin/main")
    args = ap.parse_args()
    bad = (tree() if args.tree else []) + (added(args.base) if args.added else [])
    for b in bad:
        print(f"refused: {b}")
    if not (args.tree or args.added):
        ap.error("give --tree and/or --added")
    print(f"check_public_ids: {len(bad)} refusal(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
