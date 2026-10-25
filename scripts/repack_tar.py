"""Re-pack a receipt archive with its members redacted, deterministically (docs/receipts-redaction.md).

The members go through the same transform as the tree (scripts/redact_receipt_ids.py, Redactor). The archive's own
SCIENCE-SHA256SUMS lines are re-cut for exactly the members that changed; every other member, instrument.sha256 and the
sources included, keeps its bytes. The output is deterministic: members in name order, mtime 0, uid/gid 0, empty owner
names, modes kept, and a gzip header with no name and mtime 0, so two builds from the same input are byte-identical.

    python scripts/repack_tar.py --map M IN.tar.gz OUT.tar.gz
"""
import argparse
import gzip
import hashlib
import io
import os
import re
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from redact_receipt_ids import Redactor, load_map  # noqa: E402

SUMS = re.compile(r"^([0-9a-f]{64})(\s+\*?)(.+?)\s*$")


def repack(src, dst, red):
    with tarfile.open(src) as t:
        members = [(m, t.extractfile(m).read() if m.isfile() else None) for m in t.getmembers()]
    changed = {}
    out = []
    for m, data in members:
        if data is not None and b"\0" not in data:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = None
            if text is not None:
                new = red.redact(m.name, text).encode("utf-8")
                if new != data:
                    changed[m.name] = (hashlib.sha256(data).hexdigest(), hashlib.sha256(new).hexdigest())
                    data = new
        out.append([m, data])
    for item in out:   # re-cut the archive's own SCIENCE-SHA256SUMS for the members that changed
        m, data = item
        if data is not None and os.path.basename(m.name) == "SCIENCE-SHA256SUMS":
            base = os.path.dirname(m.name)
            lines = data.decode("utf-8").split("\n")
            for i, ln in enumerate(lines):
                s = SUMS.match(ln)
                key = os.path.normpath(os.path.join(base, s.group(3))) if s else None
                if s and key in changed:
                    if s.group(1) != changed[key][0]:
                        sys.exit(f"refusing: SCIENCE-SHA256SUMS line for {key} does not hold its original sha")
                    lines[i] = f"{changed[key][1]}{s.group(2)}{s.group(3)}"
            item[1] = "\n".join(lines).encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as t:
        for m, data in sorted(out, key=lambda x: x[0].name):
            info = tarfile.TarInfo(m.name)
            info.type, info.mode, info.mtime = m.type, m.mode, 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.linkname = m.linkname
            if data is not None:
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
            else:
                t.addfile(info)
    with open(dst, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=9) as g:
            g.write(buf.getvalue())
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()
    changed = repack(args.src, args.dst, Redactor(load_map(args.map)))
    print(f"re-packed {args.dst}: {len(changed)} members redacted; "
          f"sha256 {hashlib.sha256(open(args.dst, 'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
