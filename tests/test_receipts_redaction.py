# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The rented-hardware identifier redaction (docs/receipts-redaction.md) stays complete, and its exemptions stay exact.

The public tree carries no GPU UUID and no provider instance id, except in HOLDS: files whose bytes another file pins,
or whose value a runner reads, so editing them would break a seal. Each hold keeps its sha256 here. The private-marker
guard skips a hold only through an `#exempt-fixture:` pin, and only the holds that carry the UUID shape need one; this
file checks the guard's list against that computed set, so it can neither grow quietly nor go stale. TEST_SOURCES feed
synthetic identifiers to the code that strips them, which is their job."""
import hashlib
import importlib.util
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARKERS = ROOT / ".github" / "private-markers.txt"
MANIFEST = ROOT / "docs" / "receipts-redaction.sha256"

_spec = importlib.util.spec_from_file_location("redact_receipt_ids", ROOT / "scripts" / "redact_receipt_ids.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

HOLDS = {   # path -> (why it keeps its identifiers, its sha256)
    "bench/dq11/DQ11-AMENDMENT-3.md": ("sealed: science.sha256 pins it", "f79c920cf55bed0c354d05c48c90591e2efad59e7e500041ba19468fcc1cc149"),
    "bench/dq11/DQ11-AMENDMENT-7.md": ("sealed: science.sha256 pins it", "f4990bdc945ca91ac2ef1e7474874b047705a22283bddbe981102a7a684e516b"),
    "bench/dq7/receipts/dq7-5090-1/read-llama31_8b-device-2048.json": ("pinned: d7-unlicensed.json records its sha256", "9433bd8cee23e0e9b5011e696841693b732a5d49d8fa08d1125f76e45d583752"),
    "bench/dq7/receipts/dq7-5090-1/read-llama31_8b-device-4096.json": ("pinned: d7-unlicensed.json records its sha256", "5ae719d286bb41d87db63b05b6ed1d93bfd4a2b5a8652c16bb075376b47b4b5e"),
    "bench/dq7/receipts/dq7-5090-1/read-llama31_8b-device-512.json": ("pinned: d7-unlicensed.json records its sha256", "0440c5fbb624f363daa4b6417593c6ceb4a97874c5f3332ae0cf427f4b8f2ac9"),
    "bench/dq7/receipts/dq7-5090-1/read-llama31_8b-stream-2048.json": ("pinned: d7-unlicensed.json records its sha256", "2bdf003bfc368274c3d0215c4621f0618f396600caadef80c9784311c9458ca5"),
    "bench/dq7/receipts/dq7-5090-1/read-llama31_8b-stream-4096.json": ("pinned: d7-unlicensed.json records its sha256", "df5efad49560b805ff25a90e9a330c2fa7e1d6702484f784f44eeafb4c798c0a"),
    "bench/dq7/receipts/dq7-5090-1/read-llama31_8b-stream-512.json": ("pinned: d7-unlicensed.json records its sha256", "3fb1da22b305edc24ab576e0011c0abb5c6b45e3d37c89ff5f04c400551326ea"),
    "bench/dq7/receipts/dq7-5090-1/read-qwen3_14b-device-2048.json": ("pinned: d7-unlicensed.json records its sha256", "5488a07f2e46bd8ca519135c834e0f52f1e500cb33ad0e3ae46c8a09d524074d"),
    "bench/dq7/receipts/dq7-5090-1/read-qwen3_14b-device-4096.json": ("pinned: d7-unlicensed.json records its sha256", "dcf1c95d7805d377ad755872ec28289c98e2826904a4d0721e3e4006434a8e8b"),
    "bench/dq7/receipts/dq7-5090-1/read-qwen3_14b-device-512.json": ("pinned: d7-unlicensed.json records its sha256", "1ce29102ec421ef433dde3db14b4ec983eb9778a6ff2ec48de38d1148270b637"),
    "bench/dq7/receipts/dq7-5090-1/read-qwen3_14b-stream-2048.json": ("pinned: d7-unlicensed.json records its sha256", "dbbed10dec78b94e50d316d5ae8d34d8a3995d92cf19479c198b06ad636e80c8"),
    "bench/dq7/receipts/dq7-5090-1/read-qwen3_14b-stream-4096.json": ("pinned: d7-unlicensed.json records its sha256", "6dd58c7028aa3f09506454466017eaa7bf9239bbe400e647524d708305e95de7"),
    "bench/dq7/receipts/dq7-5090-1/read-qwen3_14b-stream-512.json": ("pinned: d7-unlicensed.json records its sha256", "ce1107f0c2f5971052bf83d1728fdf25963e5ecde2c3e23282811e0d1daaf0e4"),
    "bench/dq7/receipts/dq7-5090-1/read-qwen3_32b-device-2048.json": ("pinned: d7-unlicensed.json records its sha256", "c7d1118086178a47311bd01bbf9fc63f908da6d04483eb773d75eba7795c7280"),
    "bench/dq7/receipts/dq7-5090-1/read-qwen3_32b-stream-2048.json": ("pinned: d7-unlicensed.json records its sha256", "e4d78ba425a696ac13a18d1b7dcfecf1f9e1d167dd83650e853b81c4851df7c8"),
    "bench/p132/PREREG-p132.md": ("P132's card reference (rc 18)", "99df83775d1c538b8007acae6ed8f2668b6c42903a9009717d2c5f0d7c8d62fd"),
    "bench/p132/p132_run.sh": ("P132's card reference (rc 18); staged-p132.sha256 pins it", "73100b5b540a41e4d1ff9e908a8f932b61b411975962ace606ae2073f2fa8c0b"),
    "bench/sd2/receipts/sd2-5090-3/forensics.txt": ("P132's card reference; test_p132_card.py pins it", "49acadba5416519b2d841955f555c67368f7172721aeccc950f537026ac4996a"),
    "bench/train-parity-20260905/tp1/gemma4_train_batched.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "95c1c22a97e583ea94ad38df4247cc784fc1d24e70fbd3935e089d491cb87761"),
    "bench/train-parity-20260905/tp1/gemma4_train_fused.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "ebbdd010d17486c6fbe3d26626e62ec4639267eb1d7b376cd31bb4631fe7780b"),
    "bench/train-parity-20260905/tp1/gemma4_train_reference.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "a6b8f6d46dbf4695c51840cf7d6ff8e5fb797de5dbd62baa4709076b7986c0df"),
    "bench/train-parity-20260905/tp1/granite_train_batched.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "f81fca5977cd05f0523050cb7c300178b8e707d44a647071fd625d3c691e1a92"),
    "bench/train-parity-20260905/tp1/granite_train_fused.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "caf53cb09379ced5d1809312c6f2e27cfe13e59b40897764b0136fd158bab344"),
    "bench/train-parity-20260905/tp1/granite_train_reference.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "1a1285859e8556968abf0cbe7540c2ad11233f8b48c416fb27e46d44e9c9c9f7"),
    "bench/train-parity-20260905/tp1/mixtral_train_batched.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "463dbb03a8bc466a2c4f1976c4d1da6aa27d88db146be8e6dcc1522785aa33e5"),
    "bench/train-parity-20260905/tp1/mixtral_train_fused.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "0883af480141f03ebc51af3abb38cfa92fd26e0c97fe5808718480a56bde7bae"),
    "bench/train-parity-20260905/tp1/mixtral_train_reference.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "efa43ba8875eddb4cdb823eb9368330c3e11eb6ed646442fc3297f9752f6649c"),
    "bench/train-parity-20260905/tp1/olmoe_train_batched.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "b68ec5f2a1ed95d01a247aa2296b3e3db1d294ea5988cd935f638d3f9e00b150"),
    "bench/train-parity-20260905/tp1/olmoe_train_fused.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "46be3798c830b57f5e767d0b38f59a19c10ae1fe70aa657eeb884bd695c971fc"),
    "bench/train-parity-20260905/tp1/olmoe_train_reference.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "86d197d7ff32f54a278e6b4a02cf247267f377ec60f2fb99c3379e4a2de3168a"),
    "bench/train-parity-20260905/tp1/qwen3_train_batched.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "348ed78d7fe62ab83a2b9f303593760e01d7e7609d03577758dd0033955a7760"),
    "bench/train-parity-20260905/tp1/qwen3_train_fused.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "49b03967e5c2411906223f27d29eed1db5567de082cb1be1eb9e3d4fb19f759d"),
    "bench/train-parity-20260905/tp1/qwen3_train_reference.json": ("pinned: bench/p67/reread-existing/reread.json records its sha256", "9ec88062c7b6fdc3d620cff83257987930f4d6b9b3f2a5374b688c9914d7507f"),
    "tests/fixtures/dq11-census-stalled/receipt.json": ("pinned: test_dq11_l_census_launch.py records its sha256", "0ff76566d4246bb78e7c0a89a57142a9a92eeb33a72255c59725bb53b88788aa"),
    "tests/fixtures/p132_canary/forensics.txt": ("P132's card reference; test_p132_card.py pins it", "8937620ac7926dec900fa58d1fe2c6978fcadcc953b2c45bc51c54733dc450eb"),
    "tests/test_p132_card.py": ("P132's card reference (rc 18)", "c3f1c86f084163c222fed847d1889a6f23785146fb84c475c99d3f76982a4ea7"),
}
TEST_SOURCES = {
    "tests/test_dq11_l_census_launch.py",
    "tests/test_drive_ssh_known_hosts.py",
    "tests/test_gpu_contract.py",
    "tests/test_ho2_public.py",
    "tests/test_ho3_public.py",
    "tests/test_p41_drivers.py",
    "tests/test_tc1_public.py",
}
LEDGER_FIXTURES = {   # the exemptions that predate the redaction: ledger rows the DQ11 census tests parse
    "tests/fixtures/dq11-census-refused/canonical-ledger.jsonl",
    "tests/fixtures/dq11-census-refused/ledger.jsonl",
    "tests/fixtures/dq11-census-stalled/ledger.jsonl",
}
GUARD_FILES = {".github/private-markers.txt", ".github/workflows/private-marker-guard.yml"}
UUID_SHAPE = re.compile(r"GPU-" + R.HEX_UUID + "|" + R.BARE_UUID_RE.pattern, re.I)


def _exempt():
    out = {}
    for ln in MARKERS.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#exempt-fixture: "):
            _, fx, cap, pin = ln.split()
            out[fx] = (int(cap), pin)
    return out


def test_every_hold_keeps_its_bytes():
    for fx, (why, pin) in HOLDS.items():
        assert hashlib.sha256((ROOT / fx).read_bytes()).hexdigest() == pin, f"{fx} changed ({why}): re-rule it, never re-pin it"


def test_the_guard_exempts_exactly_the_holds_that_carry_a_uuid():
    carry = {fx for fx in HOLDS if UUID_SHAPE.search((ROOT / fx).read_text(encoding="utf-8"))}
    exempt = _exempt()
    assert set(exempt) == carry | LEDGER_FIXTURES
    for fx in carry:
        assert exempt[fx][1] == HOLDS[fx][1], f"{fx}: the guard's pin is not the hold's"


def test_no_identifier_outside_the_holds():
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    stray = {}
    for fx in tracked.decode("utf-8").split("\0"):
        if not fx or fx in HOLDS or fx in TEST_SOURCES or fx in GUARD_FILES:
            continue
        try:
            text = (ROOT / fx).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        found = R.find_values(text)
        if found["gpu"] or found["instance"]:
            stray[fx] = len(found["gpu"]) + len(found["instance"])
    assert not stray, f"identifiers outside the holds; run scripts/redact_receipt_ids.py: {dict(list(stray.items())[:5])}"


def test_the_manifest_matches_the_tree():
    rows = [ln.split("  ", 3) for ln in MANIFEST.read_text(encoding="utf-8").splitlines() if ln and ln[0] != "#"]
    assert len(rows) == 2155
    for orig, red, origin, fx in rows:
        assert origin in ("store", "git:605d21e6"), origin
        assert orig != red and fx not in HOLDS and fx not in TEST_SOURCES
        assert hashlib.sha256((ROOT / fx).read_bytes()).hexdigest() == red, f"{fx} is not its redacted bytes"


def test_tokens_keep_identity_and_lose_the_shape():
    key = b"k" * 32
    one = "GPU-" + "-".join(("1" * 8, "2" * 4, "3" * 4, "4" * 4, "5" * 12))   # built here so this file names none
    two = "GPU-" + "-".join(("6" * 8, "7" * 4, "8" * 4, "9" * 4, "a" * 12))
    inst = str(7 * 10 ** 7 + 4321)
    text = R.redact(f'{one} {one} {two} "vast_instance_id": {inst} Vast instance {inst}', key)
    assert R.find_values(text) == {"gpu": set(), "instance": set()}
    toks = re.findall(r"GPU-redacted-([0-9a-f]{8})", text)
    assert len(toks) == 3 and toks[0] == toks[1] != toks[2]          # equal stays equal, different stays different
    assert '"vast_instance_id": "instance-redacted-' in text          # a JSON number becomes a string token
    zero = "GPU-" + "-".join(("0" * 8, "0" * 4, "0" * 4, "0" * 4, "0" * 12))
    assert R.redact(zero, key) == zero                                # the guard's canary stays


# ---- the claims register: a redaction of a host descriptor, never a change of claim ----
# "Claims are never edited, only superseded" protects a claim's substance: its number, meaning, unit, status, tier and
# evidence. The redaction touched only the descriptive text fields below, and only by the instance-id substitution.
TEXT_FIELDS = ("hardware", "notes", "claim")
ORIGIN = "605d21e6"   # v0.58.1, the originals


def claims_problems(old, new):
    """Every way the branch's register differs from v0.58.1 other than the instance-id substitution."""
    out = []
    o, n = old["claims"], new["claims"]
    if [c["id"] for c in o] != [c["id"] for c in n]:
        return ["the claim ids or their order changed"]
    if {k: v for k, v in old.items() if k != "claims"} != {k: v for k, v in new.items() if k != "claims"}:
        out.append("a top-level field changed")
    tokens = {}
    for a, b in zip(o, n):
        for k in set(a) | set(b):
            if k not in TEXT_FIELDS:
                if a.get(k) != b.get(k):
                    out.append(f"{a['id']}: {k} changed")
                continue
            x, y = a.get(k), b.get(k)
            if x == y:
                continue
            if not (isinstance(x, str) and isinstance(y, str)):
                out.append(f"{a['id']}: {k} changed type")
                continue
            # the old text, with each instance id it names turned into a token slot, must match the new text exactly
            parts, last, ids = [], 0, []
            for m in R.INSTANCE_PROSE_RE.finditer(x):
                parts += [re.escape(x[last:m.start(2)]), "redacted-([0-9a-f]{8})"]
                ids.append(m.group(2))
                last = m.end(2)
            parts.append(re.escape(x[last:]))
            got = re.fullmatch("".join(parts), y)
            if not ids or not got:
                out.append(f"{a['id']}: {k} differs by more than the instance-id substitution")
                continue
            for value, tok in zip(ids, got.groups()):
                if tokens.setdefault(value, tok) != tok:
                    out.append(f"{a['id']}: one instance id took two tokens")
    return out


def _registers():
    import json
    old = json.loads(subprocess.run(["git", "-C", str(ROOT), "show", f"{ORIGIN}:docs/claims.json"],
                                    capture_output=True, check=True).stdout)
    return old, json.loads((ROOT / "docs" / "claims.json").read_text(encoding="utf-8"))


def test_the_register_changed_only_by_the_instance_id_substitution():
    old, new = _registers()
    assert claims_problems(old, new) == []
    assert sum(a != b for a, b in zip(old["claims"], new["claims"])) == 353            # the rows the redaction touched


def test_a_changed_digit_in_a_value_fails_the_register_check():
    import copy
    old, new = _registers()
    bad = copy.deepcopy(new)
    row = next(c for c in bad["claims"] if isinstance(c.get("value"), (int, float)))
    row["value"] = row["value"] + 1
    assert any("value changed" in p for p in claims_problems(old, bad))
    bad = copy.deepcopy(new)
    row = next(c for c in bad["claims"] if "redacted-" in c.get("hardware", ""))
    row["hardware"] = row["hardware"].replace("RTX", "GTX", 1)
    assert any("more than the instance-id substitution" in p for p in claims_problems(old, bad))
