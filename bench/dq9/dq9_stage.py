"""Derive the flat stage list from registered checksums; refuse missing, changed or colliding subjects."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


def stage(directory):
    here = Path(directory)
    root = here.parents[1]
    sources = (here, root/"bench/dq7", root/"bench/dq7/configs", root/"bench/dq3")
    manifest = here/"instrument.sha256"
    names, paths = set(), []
    for line in manifest.read_text().splitlines():
        digest, name = line.split()
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or Path(name).name != name or name in names:
            raise ValueError("invalid or colliding checksum subject: "+name)
        names.add(name)
        matching = [source/name for source in sources if (source/name).is_file()
                    and hashlib.sha256((source/name).read_bytes()).hexdigest() == digest]
        if not matching:
            raise ValueError("checksum subject absent or changed: "+name)
        paths.append(matching[0])
    if not paths or "instrument.sha256" in names:
        raise ValueError("empty or self-referential checksum manifest")
    return (*paths, manifest)


if __name__ == "__main__":
    print(" ".join(str(p) for p in stage(Path(__file__).resolve().parent)))
