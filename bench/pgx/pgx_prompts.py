"""PGX scored prompt sets: three sets of 1,024-token raw-text windows, cut at fixed offsets from four pinned sources.

    python bench/pgx/pgx_prompts.py --fetch SRC_DIR                         # the pinned source files (~84 MB)
    python bench/pgx/pgx_prompts.py --build SRC_DIR --tokenizer tokenizer.json --out bench/pgx/prompts

Sources, one per content kind, each pinned by revision (the licence lines are in ``SOURCES``):
  prose     wikitext-2-raw-v1 test: non-empty rows joined by blank lines (SC2's corpus rendering)
  code      CPython ``Lib/`` files at v3.13.9 (``CODE_FILES``, in that order) joined by blank lines
  math      hendrycks_math test, subjects in sorted order: each row "Problem: <problem>\\n\\nSolution: <solution>"
  dialogue  ultrachat_200k test_sft, file order: each conversation's turns as "User: ..." / "Assistant: ..." lines.
            Raw text, NOT a chat template, so every engine is sent the same token ids.

Each kind's corpus is tokenized once with the shared Qwen3-30B-A3B tokenizer (``tokenizer.json`` sha256 prefix
``aeb13307``, no special tokens added) and window i is ids[i * STRIDE : i * STRIDE + 1024]. The windows are assigned
in index order:
  warm_c1 (P1, P2):  code 0-3,  dialogue 0-3    (8, interleaved code, dialogue, code, ...)
  warm_c4 (P3):      code 4-11, dialogue 4-11   (16, interleaved)
  shift   (P4):      math 0-3,  prose 0-3       (8, interleaved)
so no window is in two sets. Each set file is the driver's prompt list ({id, kind, token_ids}) plus the window's
decoded ``text`` (for the kit test's disjointness check; the driver sends only the ids). ``manifest.json`` records
every source file's sha256, the tokenizer's, the offsets and each set's ``prompts_sha`` (``pgx_drive.prompts_sha``).

The build refuses: a source file whose sha256 differs from the one recorded at fetch, a corpus too short for its
windows, a window that is not exactly 1,024 tokens, and any window containing text of the four #1469 capture prompts
(``capture_prompts.json``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import pgx_drive  # noqa: E402

WINDOW = 1024
STRIDE = 8192
TOKENIZER_SHA = "aeb13307"
CPYTHON = ("v3.13.9", "8183fa5e3f78ca6ab862de7fb8b14f3d929421e0")
CODE_FILES = ("Lib/argparse.py", "Lib/dataclasses.py", "Lib/functools.py", "Lib/inspect.py", "Lib/typing.py",
              "Lib/collections/__init__.py", "Lib/pathlib/_local.py", "Lib/enum.py", "Lib/textwrap.py",
              "Lib/difflib.py")
MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                 "prealgebra", "precalculus")
SOURCES = {
    "prose": {"repo": "Salesforce/wikitext", "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
              "files": ("wikitext-2-raw-v1/test-00000-of-00001.parquet",), "license": "CC-BY-SA-3.0 / GFDL"},
    "math": {"repo": "EleutherAI/hendrycks_math", "revision": "21a5633873b6a120296cce3e2df9d5550074f4a3",
             "files": tuple(f"{s}/test-00000-of-00001.parquet" for s in MATH_SUBJECTS), "license": "MIT"},
    "dialogue": {"repo": "HuggingFaceH4/ultrachat_200k", "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
                 "files": ("data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",), "license": "MIT"},
    "code": {"repo": "python/cpython", "revision": CPYTHON[1], "tag": CPYTHON[0], "files": CODE_FILES,
             "license": "PSF-2.0"},
}
# kind -> window indices per set; sets interleave their two kinds in index order
SETS = {"warm_c1": (("code", range(0, 4)), ("dialogue", range(0, 4))),
        "warm_c4": (("code", range(4, 12)), ("dialogue", range(4, 12))),
        "shift": (("math", range(0, 4)), ("prose", range(0, 4)))}


def _url(kind: str, path: str) -> str:
    s = SOURCES[kind]
    if kind == "code":
        return f"https://raw.githubusercontent.com/{s['repo']}/{s['revision']}/{path}"
    return f"https://huggingface.co/datasets/{s['repo']}/resolve/{s['revision']}/{path}"


def _local(src: str, kind: str, path: str) -> str:
    return os.path.join(src, kind, path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fetch(src: str) -> dict:
    shas = {}
    for kind, s in SOURCES.items():
        for path in s["files"]:
            dst = _local(src, kind, path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            req = urllib.request.Request(_url(kind, path), headers={"User-Agent": "pgx-prompts"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dst + ".part", "wb") as f:
                while blk := r.read(1 << 20):
                    f.write(blk)
            os.replace(dst + ".part", dst)
            shas[f"{kind}/{path}"] = sha256_file(dst)
    with open(os.path.join(src, "fetched.json"), "w", encoding="utf-8") as f:
        json.dump(shas, f, indent=1, sort_keys=True)
    return shas


def _rows(path: str, cols):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=list(cols))
    return t.to_pylist()


def render(src: str) -> dict:
    """kind -> the corpus text, rendered as the module docstring states."""
    out = {}
    out["prose"] = "\n\n".join(r["text"] for r in _rows(_local(src, "prose", SOURCES["prose"]["files"][0]), ["text"])
                               if r["text"].strip())
    math = []
    for path in SOURCES["math"]["files"]:
        math += [f"Problem: {r['problem']}\n\nSolution: {r['solution']}"
                 for r in _rows(_local(src, "math", path), ["problem", "solution"])]
    out["math"] = "\n\n".join(math)
    convs = []
    for r in _rows(_local(src, "dialogue", SOURCES["dialogue"]["files"][0]), ["messages"]):
        convs.append("\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                               for m in r["messages"]))
    out["dialogue"] = "\n\n".join(convs)
    code = []
    for path in CODE_FILES:
        with open(_local(src, "code", path), encoding="utf-8") as f:
            code.append(f.read())
    out["code"] = "\n\n".join(code)
    return out


def capture_prompts() -> dict:
    with open(os.path.join(_HERE, "capture_prompts.json"), encoding="utf-8") as f:
        return json.load(f)["prompts"]


def capture_fragments(min_len: int = 24) -> list:
    """Every line (and every sentence) of the capture prompts at least ``min_len`` characters long."""
    frags = set()
    for text in capture_prompts().values():
        for piece in text.replace(". ", ".\n").split("\n"):
            piece = piece.strip()
            if len(piece) >= min_len:
                frags.add(piece)
    return sorted(frags)


def windows(ids: list, n: int) -> list:
    need = (n - 1) * STRIDE + WINDOW
    if len(ids) < need:
        raise SystemExit(f"corpus of {len(ids)} tokens is short of the {need} its {n} windows need")
    return [ids[i * STRIDE: i * STRIDE + WINDOW] for i in range(n)]


def build(src: str, tokenizer_path: str, out_dir: str) -> dict:
    with open(os.path.join(src, "fetched.json"), encoding="utf-8") as f:
        fetched = json.load(f)
    for key, want in fetched.items():
        kind, path = key.split("/", 1)
        if sha256_file(_local(src, kind, path)) != want:
            raise SystemExit(f"{key} changed since the fetch")
    tok_sha = sha256_file(tokenizer_path)
    if not tok_sha.startswith(TOKENIZER_SHA):
        raise SystemExit(f"tokenizer sha256 {tok_sha[:8]} is not the shared {TOKENIZER_SHA}")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    corpora = render(src)
    need = {k: 0 for k in SOURCES}
    for parts in SETS.values():
        for kind, idx in parts:
            need[kind] = max(need[kind], max(idx) + 1)
    wins = {}
    for kind, text in corpora.items():
        ids = tok.encode(text, add_special_tokens=False).ids
        wins[kind] = windows(ids, need[kind])
    frags = capture_fragments()
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"window": WINDOW, "stride": STRIDE, "tokenizer_sha256": tok_sha,
                "sources": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in s.items()}
                            for k, s in SOURCES.items()},
                "fetched_sha256": fetched, "sets": {}}
    for name, parts in SETS.items():
        (ka, ia), (kb, ib) = parts
        order = [x for pair in zip([(ka, i) for i in ia], [(kb, i) for i in ib]) for x in pair]
        prompts = []
        for kind, i in order:
            w = wins[kind][i]
            text = tok.decode(w, skip_special_tokens=False)
            if len(w) != WINDOW:
                raise SystemExit(f"{kind} window {i} is {len(w)} tokens")
            hit = [fr for fr in frags if fr in text]
            if hit:
                raise SystemExit(f"{kind} window {i} contains capture-prompt text: {hit[0]!r}")
            prompts.append({"id": f"{name}-{kind}-{i}", "kind": kind, "window": i, "token_ids": w, "text": text})
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(prompts, f, separators=(",", ":"))
            f.write("\n")
        manifest["sets"][name] = {"n": len(prompts), "prompts_sha": pgx_drive.prompts_sha(prompts),
                                  "kinds": sorted({p["kind"] for p in prompts})}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", metavar="SRC_DIR")
    ap.add_argument("--build", metavar="SRC_DIR")
    ap.add_argument("--tokenizer")
    ap.add_argument("--out", default=os.path.join(_HERE, "prompts"))
    a = ap.parse_args(argv)
    if a.fetch:
        for k, v in fetch(a.fetch).items():
            print(f"{v[:12]}  {k}")
    if a.build:
        m = build(a.build, a.tokenizer, a.out)
        for name, s in m["sets"].items():
            print(f"{name:8s} n={s['n']:2d} prompts_sha {s['prompts_sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
