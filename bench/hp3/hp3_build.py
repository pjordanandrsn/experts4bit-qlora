#!/usr/bin/env python3
"""bench/hp3/hp3_build.py -- HP3's resolver check (bench/hp3/HP3-PREREG.md, "The matched build").

`probe --framework F --out P.json` (run with each venv's own interpreter): what that venv actually installed. That is torch (and its
CUDA), bitsandbytes and the sha256 of the libbitsandbytes it loads, triton, transformers, peft and the framework's own version.

`compare P1.json P2.json ... --want-torch V --want-bnb V --out build.json`: the build fields must be the same in every venv and must
equal the pins. Those fields are torch, CUDA, bitsandbytes, the bnb library's sha256 and triton. transformers and peft are each
framework's own, recorded and never compared. Exit 0 MATCHED, 5 MISMATCH (the (R) fallback signal under the shakedown token), 2 unusable.

`classify --log L [--receipt R] --vram-mib N --floor-mib M` (ruling 5): a shakedown arm that installed and imported but did not run.
The class is read from the error text the arm left, never set by hand:
- HOST_OOM_UNDER_FLOOR: CUDA out of memory on a card whose VRAM is under the registered floor;
- HOST_DRIVER: a driver, Xid or ECC error;
- HOST_PREEMPT: provider preemption or teardown mid-run.
Those three use the spare. Anything else is R, the (R) trigger. Prints "<class> | <the quoted error line>".
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import re
import sys

BUILD_FIELDS = ("torch", "cuda", "bitsandbytes", "bnb_lib_sha256", "triton")
EXIT_MATCHED, EXIT_UNUSABLE, EXIT_MISMATCH = 0, 2, 5
FRAMEWORK_DIST = {"e4b": "experts4bit-qlora", "unsloth": "unsloth", "axolotl": "axolotl"}


def _ver(dist):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


FRAMEWORK_IMPORTS = {"e4b": ("experts4bit_qlora", "nf4_qlora"), "unsloth": ("unsloth", "unsloth_zoo"),
                     "axolotl": ("axolotl", "axolotl.loaders")}


def probe(framework):
    import importlib
    for m in FRAMEWORK_IMPORTS[framework]:      # the framework first: Unsloth patches transformers at import
        importlib.import_module(m)
    import bitsandbytes
    import torch
    if not torch.cuda.is_available():
        raise ImportError(f"no CUDA in the {framework} venv")
    rec = {"framework": framework, "framework_version": _ver(FRAMEWORK_DIST[framework]), "torch": torch.__version__,
           "cuda": torch.version.cuda, "bitsandbytes": bitsandbytes.__version__, "triton": _ver("triton"),
           "transformers": _ver("transformers"), "peft": _ver("peft"), "python": sys.version.split()[0]}
    if framework == "unsloth":
        rec["unsloth_zoo"] = _ver("unsloth_zoo")
    path = bnb_lib_path()
    rec["bnb_lib"] = path.rsplit("/", 1)[-1] if path else None
    rec["bnb_lib_sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest() if path else None
    return rec


def bnb_lib_path(maps=None):
    """The libbitsandbytes*.so mapped into this process (as tc1_arm's hbd_libraries reads it): None if none or several differ."""
    if maps is None:
        with open("/proc/self/maps") as f:
            maps = f.read()
    paths = set()
    for ln in maps.splitlines():
        parts = ln.split()
        if len(parts) >= 6 and "libbitsandbytes" in parts[-1].rsplit("/", 1)[-1]:
            paths.add(parts[-1])
    return paths.pop() if len(paths) == 1 else None


def compare(probes, want_torch, want_bnb):
    """The build.json record. A field that differs across venvs, or a pin that is not met, is a MISMATCH naming it."""
    bad = []
    for f in BUILD_FIELDS:
        vals = {p["framework"]: p.get(f) for p in probes}
        if len(set(vals.values())) != 1 or None in vals.values():
            bad.append(f"{f} differs across venvs: {vals}")
    for p in probes:
        if (p.get("torch") or "").split("+")[0] != want_torch:
            bad.append(f"{p['framework']}: torch {p.get('torch')} != the pin {want_torch}")
        if not (p.get("torch") or "").endswith("+cu130"):
            bad.append(f"{p['framework']}: torch {p.get('torch')} is not a cu130 build")
        if p.get("bitsandbytes") != want_bnb:
            bad.append(f"{p['framework']}: bitsandbytes {p.get('bitsandbytes')} != the pin {want_bnb}")
    return {"tool": "bench/hp3/hp3_build.py compare", "build_fields": list(BUILD_FIELDS), "want": {"torch": want_torch, "bitsandbytes": want_bnb},
            "venvs": {p["framework"]: p for p in probes},
            "reported_never_compared": {p["framework"]: {"transformers": p.get("transformers"), "peft": p.get("peft")} for p in probes},
            "verdict": "MISMATCH" if bad else "MATCHED", "problems": bad}


HOST_CLASSES = (
    ("HOST_DRIVER", re.compile(r"\bXid\b|\bECC\b|uncorrectable|GPU is lost|fallen off the bus|fell off the bus|NVML|"
                               r"CUDA driver version is insufficient|no CUDA-capable device|cudaErrorDevicesUnavailable|unspecified launch failure "
                               r"\(driver\)", re.I)),
    ("HOST_PREEMPT", re.compile(r"\bpreempted\b|\bpreemption\b|instance (?:is |was )?(?:being )?(?:stopped|terminated|reclaimed)|spot (?:instance )?interrupt", re.I)),
)
OOM = re.compile(r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED", re.I)
ERRLINE = re.compile(r"Error|Exception|Traceback|error:|FAILED|Killed", re.I)


def classify(text, vram_mib, floor_mib):
    """(class, the quoted line) for an arm that did not run. Host classes first, by their own text; OOM is host-attributable only on a
    card under the floor; anything else is R."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for name, pat in HOST_CLASSES:
        hit = [ln for ln in lines if pat.search(ln)]
        if hit:
            return name, hit[-1][:300]
    oom = [ln for ln in lines if OOM.search(ln)]
    if oom and vram_mib is not None and vram_mib < floor_mib:
        return "HOST_OOM_UNDER_FLOOR", oom[-1][:300]
    err = [ln for ln in lines if ERRLINE.search(ln)]
    return "R", ((oom or err or lines or ["(no error text captured)"])[-1])[:300]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("probe")
    a1.add_argument("--framework", choices=sorted(FRAMEWORK_DIST), required=True)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("compare")
    a2.add_argument("probes", nargs="+")
    a2.add_argument("--want-torch", required=True)
    a2.add_argument("--want-bnb", required=True)
    a2.add_argument("--out", required=True)
    a3 = sub.add_parser("classify")
    a3.add_argument("--log", required=True)
    a3.add_argument("--receipt")
    a3.add_argument("--vram-mib", type=int)
    a3.add_argument("--floor-mib", type=int, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "classify":
        text = ""
        try:
            text = open(a.log, errors="replace").read() + "\n"
        except OSError:
            pass
        if a.receipt:
            try:
                text += str(json.load(open(a.receipt)).get("reason") or "") + "\n"
            except (OSError, ValueError, AttributeError):
                pass
        cls, line = classify(text, a.vram_mib, a.floor_mib)
        print(f"{cls} | {line}")
        return EXIT_MATCHED
    try:
        if a.cmd == "probe":
            rec = probe(a.framework)
            json.dump(rec, open(a.out, "w"), indent=1, sort_keys=True)
            print(f"HP3_BUILD_PROBE {a.framework}: torch {rec['torch']} bnb {rec['bitsandbytes']} lib {rec['bnb_lib']} triton {rec['triton']}")
            return EXIT_MATCHED
        rec = compare([json.load(open(p)) for p in a.probes], a.want_torch, a.want_bnb)
        json.dump(rec, open(a.out, "w"), indent=1, sort_keys=True)
        print(f"HP3_BUILD={rec['verdict']}" + (f" {rec['problems'][:3]}" if rec["problems"] else ""))
        return EXIT_MATCHED if rec["verdict"] == "MATCHED" else EXIT_MISMATCH
    except (ValueError, KeyError, OSError, ImportError, json.JSONDecodeError) as e:
        print(f"HP3_BUILD UNUSABLE {type(e).__name__}: {e}")
        return EXIT_UNUSABLE


if __name__ == "__main__":
    sys.exit(main())
