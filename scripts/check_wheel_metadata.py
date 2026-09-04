#!/usr/bin/env python3
"""Assert the built wheel's METADATA carries the intended PyPI identity.

    python -m build && python scripts/check_wheel_metadata.py dist/*.whl

Checks (standard library only): Name equals pyproject's name; Summary is
the pyproject description; Keywords present; Requires-Python present;
License-Expression is MIT (PEP 639) and License-File names LICENSE (and
every file in ``license-files``); the well-known Project-URL labels are
present; every declared extra is present as Provides-Extra; and the
canonical dependency relationship named in --requires is in Requires-Dist.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path


def _pyproject(root: Path) -> dict:
    try:
        import tomllib
        return tomllib.loads((root / "pyproject.toml").read_text())
    except ImportError:
        print("check_wheel_metadata needs Python >= 3.11 (tomllib)")
        sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wheel")
    ap.add_argument("--root", default=".")
    ap.add_argument("--url-labels", default="Homepage,Documentation,Source,Issues,Changelog,Release Notes,Status,Capabilities,Solutions,Benchmarks")
    ap.add_argument("--requires", action="append", default=[], help="a Requires-Dist prefix that must be present, e.g. 'grouped-nf4-gemm>=0.28.0; extra == \"fast\"'")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    py = _pyproject(root)["project"]
    with zipfile.ZipFile(a.wheel) as z:
        meta_name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        meta = z.read(meta_name).decode()
        names = z.namelist()
    head = meta.split("\n\n", 1)[0]
    fields: dict[str, list[str]] = {}
    for line in head.splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            fields.setdefault(k.strip(), []).append(v.strip())
    errors = []
    def one(k):
        return (fields.get(k) or [""])[0]
    if one("Name") != py["name"]:
        errors.append(f"Name {one('Name')!r} != pyproject {py['name']!r}")
    if one("Summary") != py.get("description", ""):
        errors.append(f"Summary differs from pyproject description: {one('Summary')!r}")
    if not one("Keywords"):
        errors.append("Keywords missing")
    if not one("Requires-Python"):
        errors.append("Requires-Python missing")
    lic = one("License-Expression")
    if lic != "MIT":
        errors.append(f"License-Expression {lic!r} != 'MIT' (PEP 639; needs setuptools >= 77)")
    if fields.get("License") and fields["License"][0]:
        errors.append(f"legacy License field still emitted: {fields['License'][0]!r}")
    if any(c.startswith("License ::") for c in fields.get("Classifier", [])):
        errors.append("deprecated 'License ::' classifier still present beside License-Expression")
    lf = set(fields.get("License-File", []))
    want_lf = set(py.get("license-files", ["LICENSE"]))
    if not want_lf <= lf:
        errors.append(f"License-File {sorted(lf)} lacks {sorted(want_lf - lf)}")
    dist_info = meta_name.rsplit("/", 1)[0]
    for f in want_lf:
        if f"{dist_info}/licenses/{f}" not in names and f"{dist_info}/{f}" not in names:
            errors.append(f"license file {f} not inside the wheel's dist-info")
    labels = {u.split(",")[0].strip() for u in fields.get("Project-URL", [])}
    for lab in [x.strip() for x in a.url_labels.split(",") if x.strip()]:
        if lab not in labels:
            errors.append(f"Project-URL label missing: {lab}")
    extras = set(fields.get("Provides-Extra", []))
    for ex in py.get("optional-dependencies", {}):
        if ex not in extras:
            errors.append(f"Provides-Extra missing: {ex}")
    reqs = fields.get("Requires-Dist", [])
    for want in a.requires:
        def norm(s):
            return re.sub(r"\s+", "", s)
        if not any(norm(r).startswith(norm(want)) for r in reqs):
            errors.append(f"Requires-Dist lacks {want!r}; have {reqs}")
    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print(f"OK: {Path(a.wheel).name}: Name={one('Name')} License-Expression={lic} License-File={sorted(lf)} "
          f"extras={sorted(extras)} urls={len(labels)} keywords={len(one('Keywords').split(','))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
