"""DQ11 draft flat closure: no transport, checksum or basename collisions."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


def stage(directory):
    here = Path(directory)
    sources = (here, here.parent / "dq1")
    manifest = here / "instrument.sha256"
    names, paths = set(), []
    for line in manifest.read_text().splitlines():
        digest, name = line.split()
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or Path(name).name != name or name in names:
            raise ValueError("invalid or colliding checksum subject: " + name)
        names.add(name)
        matches = [source / name for source in sources if (source / name).is_file()]
        if len(matches) != 1 or hashlib.sha256(matches[0].read_bytes()).hexdigest() != digest:
            raise ValueError("checksum subject absent, ambiguous or changed: " + name)
        paths.append(matches[0])
    if not paths or "instrument.sha256" in names:
        raise ValueError("empty or self-referential checksum manifest")
    return (*paths, manifest)


if __name__ == "__main__":
    print(" ".join(str(p) for p in stage(Path(__file__).resolve().parent)))
