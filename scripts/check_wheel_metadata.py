#!/usr/bin/env python3
"""Assert the built wheel's METADATA carries the intended PyPI identity.

    python -m build && python scripts/check_wheel_metadata.py dist/*.whl

Checks (standard library only; METADATA parsed with email.parser so folded
continuation lines are handled): Name equals pyproject's name; Summary is the
pyproject description; Keywords present; Requires-Python present;
License-Expression equals pyproject's ``license`` (PEP 639) and every
``license-files`` glob matches both a License-File header and a file under
dist-info/licenses/; every [project.urls] label is a Project-URL (plus any
extra --url-labels) and no Project-URL label is longer than 32 characters --
PyPI rejects the upload otherwise; every declared extra is a Provides-Extra
and each of its requirements is present as
``Requires-Dist: <req>; extra == "<name>"``; and every --requires
(repeatable) prefix is in Requires-Dist.
"""
from __future__ import annotations

import argparse
import email.parser
import fnmatch
import os
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discovery_common import load_pyproject, pep503_name  # noqa: E402

#: PyPI (warehouse) rejects a Project-URL whose label exceeds this.
MAX_LABEL = 32
#: PEP 639 defaults, used only when pyproject names no license-files.
DEFAULT_LICENSE_GLOBS = ("LICEN[CS]E*", "COPYING*", "NOTICE*", "AUTHORS*")


def _fields(meta: str) -> dict[str, list[str]]:
    msg = email.parser.Parser().parsestr(meta, headersonly=True)
    out: dict[str, list[str]] = {}
    for k, v in msg.items():
        out.setdefault(k, []).append(" ".join(str(v).split()))
    return out


def _norm_req(req: str) -> str:
    """Comparison form of a requirement: PEP 503 name, no whitespace, double
    quotes and no parentheses in the marker."""
    req = req.strip()
    m = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", req)
    name = pep503_name(m.group(0)) if m else ""
    rest = req[m.end():] if m else req
    return name + re.sub(r"[\s()]+", "", rest).replace("'", '"')


def _expected_extra_req(req: str, extra: str) -> str:
    """What setuptools emits for ``req`` under ``extra``: the requirement's own
    marker, if any, and-ed with ``extra == "<name>"``."""
    spec, _, marker = req.partition(";")
    marker = marker.strip()
    full = f'{marker} and extra == "{extra}"' if marker else f'extra == "{extra}"'
    return _norm_req(f"{spec.strip()}; {full}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wheel")
    ap.add_argument("--root", default=".")
    ap.add_argument("--url-labels", default="",
                    help="comma-separated Project-URL labels required IN ADDITION to the [project.urls] keys")
    ap.add_argument("--requires", action="append", default=[],
                    help="a Requires-Dist prefix that must be present (repeatable), e.g. 'grouped-nf4-gemm>=0.28.0; extra == \"fast\"'")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    py = load_pyproject(root)
    with zipfile.ZipFile(a.wheel) as z:
        meta_name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        meta = z.read(meta_name).decode("utf-8")
        names = z.namelist()
    fields = _fields(meta)
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
    want_lic = py.get("license")
    if not isinstance(want_lic, str):
        errors.append(f"pyproject `license` is {want_lic!r}, not a PEP 639 expression string; License-Expression cannot be compared")
    elif lic != want_lic:
        errors.append(f"License-Expression {lic!r} != pyproject license {want_lic!r} (PEP 639; needs setuptools >= 77)")
    if fields.get("License") and fields["License"][0]:
        errors.append(f"legacy License field still emitted: {fields['License'][0]!r}")
    if any(c.startswith("License ::") for c in fields.get("Classifier", [])):
        errors.append("deprecated 'License ::' classifier still present beside License-Expression")
    lf = fields.get("License-File", [])
    dist_info = meta_name.rsplit("/", 1)[0]
    lic_dir = f"{dist_info}/licenses/"
    shipped = [n[len(lic_dir):] for n in names if n.startswith(lic_dir)]
    globs = py.get("license-files")
    if globs:
        for g in globs:
            if not any(fnmatch.fnmatchcase(x, g) for x in lf):
                errors.append(f"license-files pattern {g!r} matches no License-File header; have {sorted(lf)}")
            if not any(fnmatch.fnmatchcase(x, g) for x in shipped):
                errors.append(f"license-files pattern {g!r} matches nothing under {lic_dir}; have {sorted(shipped)}")
    else:
        if not any(fnmatch.fnmatchcase(x, g) for x in lf for g in DEFAULT_LICENSE_GLOBS):
            errors.append(f"no License-File matches the PEP 639 defaults {DEFAULT_LICENSE_GLOBS}; have {sorted(lf)}")
        if not any(fnmatch.fnmatchcase(x, g) for x in shipped for g in DEFAULT_LICENSE_GLOBS):
            errors.append(f"nothing under {lic_dir} matches the PEP 639 defaults; have {sorted(shipped)}")
    labels = [u.split(",", 1)[0].strip() for u in fields.get("Project-URL", [])]
    want_labels = list((py.get("urls") or {}).keys()) + [x.strip() for x in a.url_labels.split(",") if x.strip()]
    for lab in want_labels:
        if lab not in labels:
            errors.append(f"Project-URL label missing: {lab}")
    for lab in labels:
        if len(lab) > MAX_LABEL:
            errors.append(f"Project-URL label {lab!r} is {len(lab)} characters; PyPI rejects labels longer than {MAX_LABEL}")
    extras = {pep503_name(x) for x in fields.get("Provides-Extra", [])}
    reqs = fields.get("Requires-Dist", [])
    norm_reqs = {_norm_req(r) for r in reqs}
    for ex, items in (py.get("optional-dependencies") or {}).items():
        ex_norm = pep503_name(ex)
        if ex_norm not in extras:
            errors.append(f"Provides-Extra missing: {ex}")
        for req in items:
            want = _expected_extra_req(req, ex_norm)
            if want not in norm_reqs:
                have = [r for r in reqs if f'extra == "{ex_norm}"' in r]
                errors.append(f"Requires-Dist lacks {req!r} under extra {ex!r} (expected {want}); have {have}")
    for want in a.requires:
        if not any(_norm_req(r).startswith(_norm_req(want)) for r in reqs):
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
