"""PGX Z5: the registration page (bench/pgx/PREREG-pgx.md) against the kit, so neither can move without the other."""
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PGX = os.path.join(ROOT, "bench", "pgx")
sys.path.insert(0, PGX)
import pgx_bytes as pb  # noqa: E402
import pgx_place_preview as pp  # noqa: E402
import pgx_reduce as red  # noqa: E402

with open(os.path.join(PGX, "PREREG-pgx.md"), encoding="utf-8") as _f:
    PAGE = _f.read()


def _section(title: str) -> str:
    i = PAGE.index(f"## {title}")
    j = PAGE.find("\n## ", i + 1)
    return PAGE[i:j if j > 0 else None]


def test_the_k_table_on_the_page_is_the_kit_s():
    rows = re.findall(r"^\| (\d+) GB \| (\d+) \| (\d+) \| (\d+)[^|]*`E4B_PAGED_VRAM_GB=([\d.]+)`|"
                      r"^\| (\d+) GB \| (\d+) \| (\d+) \| (\d+); `([\d.]+)`", _section("The matched budget"), re.M)
    got = {}
    for r in rows:
        b, fp8, int4, nf4, gib = r[:5] if r[0] else r[5:]
        got[int(b)] = (int(fp8), int(int4), int(nf4), float(gib))
    with open(os.path.join(PGX, "bytes_per_expert.json"), encoding="utf-8") as f:
        kt = pb.k_table(json.load(f))
    want = {b: (kt["S8"][str(b)], kt["S4"][str(b)], kt["E-hyb"][str(b)], pp.vram_gib(b)) for b in pb.BUDGETS_GB}
    assert got == want
    assert kt["S8"] == kt["V8"]
    assert tuple(pb.BUDGETS_GB) == tuple(pp.BUDGETS_GB) == tuple(red.BUDGETS)


def test_the_bytes_per_expert_on_the_page_are_the_measured_ones():
    with open(os.path.join(PGX, "bytes_per_expert.json"), encoding="utf-8") as f:
        rec = json.load(f)
    budget = _section("The matched budget")
    for n in (rec["fp8"]["bytes_per_expert"], rec["int4"]["bytes_per_expert"],
              rec["int4"]["bytes_per_expert_without_g_idx"], pb.NF4_BYTES_PER_EXPERT):
        assert f"{n:,}" in budget, n
    assert pp.NF4_BYTES_PER_EXPERT == pb.NF4_BYTES_PER_EXPERT


def test_the_checkpoint_pins_on_the_page_are_the_kit_s():
    arms = _section("The arms")
    for repo, rev in pb.CHECKPOINTS.values():
        assert re.search(rf"`{re.escape(repo)}`[^|`@]*@{rev[:8]}\b", arms), repo


def test_the_fit_profile_named_on_the_page_is_the_committed_one():
    with open(os.path.join(PGX, "fit_profile.jsonl"), "rb") as f:
        blob = f.read()
    assert f"`{hashlib.sha256(blob).hexdigest()}`" in PAGE
    meta = json.loads(blob.split(b"\n", 1)[0])
    for name, sha in meta["traces"].items():
        kind = name.removeprefix("qwen3_").removesuffix(".jsonl")
        assert f"{kind} `{sha[:8]}`" in PAGE, kind


def test_the_arms_controls_passes_and_rules_on_the_page_are_the_reducer_s():
    for arm in red.ARMS:
        assert f"| **{arm}** |" in _section("The arms"), arm
    for arm, bud, passes in red.CONTROLS:
        assert f"**{arm}**" in PAGE, arm
    work = _section("Workload")
    for p, sets in red.PASS_SET.items():
        assert f"**{p} (" in work, p
        assert f"**{sets}** (" in work and p in re.search(rf"\*\*{sets}\*\* \(([^)]*)\)", work)[1], (p, sets)
    assert f"**{red.STABLE:.2f}**" in _section("Validity and the rule (`bench/pgx/pgx_reduce.py`, self-tested before data)")
    assert f"≥ {int(red.SHARE * 100)} %" in _section("Readings")
    assert red.registered_passes(8) == ("P1", "P2", "P3", "P4") and red.registered_passes(4) == ("P1", "P2", "P3")


def test_the_page_registers_no_placeholder_sha():
    """A prompts_sha is written only once computed from committed sets (the lane's standing rule)."""
    for h in re.findall(r"\b[0-9a-f]{64}\b", PAGE):
        assert len(set(h)) > 4, h
    assert not re.search(r"prompts_sha[^\n]*(TBD|TODO|xxxx|<sha|\.\.\.)", PAGE, re.I)
