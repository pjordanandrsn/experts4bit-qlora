"""Seal real DQ11 transport closure, including external canonical input bytes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from dq11_common import file_sha


def stage(directory, assets):
    here, assets = Path(directory), Path(assets)
    sources = (here, here.parent / "dq1", here.parent / "dq3")
    manifest = here / "science.sha256"
    names, paths = set(), []
    for line in manifest.read_text().splitlines():
        digest, name = line.split()
        if not re.fullmatch("[0-9a-f]{64}", digest) or Path(name).name != name or name in names:
            raise ValueError("invalid/colliding science closure")
        names.add(name)
        matches = [source / name for source in sources if (source / name).is_file()]
        if len(matches) != 1 or file_sha(matches[0]) != digest:
            raise ValueError("missing or changed science closure: " + name)
        paths.append(matches[0])
    if not paths or "science.sha256" in names:
        raise ValueError("empty/self-referential science closure")
    locked = json.loads((here / "locked_inputs.json").read_text())
    for name, row in locked["assets"].items():
        if Path(name).name != name or name in names:
            raise ValueError("unsafe/colliding canonical asset")
        matches = [p for p in (assets / name, assets / "adapters" / name, assets / "data" / name) if p.is_file()]
        if len(matches) != 1 or file_sha(matches[0]) != row["sha256"] or matches[0].stat().st_size != row["bytes"]:
            raise ValueError("missing, ambiguous or changed canonical asset: " + name)
        names.add(name)
        paths.append(matches[0])
    paths.append(manifest)
    if any(re.search(r"\s", str(path)) for path in paths):
        raise ValueError("TC1 flat transport requires paths without whitespace")
    return tuple(paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    print(" ".join(str(path) for path in stage(args.directory, args.assets)))
