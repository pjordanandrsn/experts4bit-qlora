# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The rented-hardware identifier redaction (docs/receipts-redaction.md) stays complete, and its exemptions stay exact.

The public tree carries no GPU UUID (or a prefix of one) and no provider instance id, except in HOLDS: files whose bytes
another file pins, or whose value a runner reads, so editing them would break a seal. Each hold keeps its sha256 here.
The private-marker guard skips a hold only through an `#exempt-fixture:` pin, and only the holds that carry the UUID
shape need one; this file checks the guard's list against that computed set, so it can neither grow quietly nor go
stale. TEST_SOURCES feed synthetic identifiers to the code that strips them, which is their job.

The completeness check here does NOT use the transform's own matching. The first redaction's check reused the
transform's finder, so it could only confirm what the transform already knew. This one judges by shape and context,
over raw bytes and gzip members, and every exception is named in ALLOWED with its count and reason."""
import collections
import copy
import gzip
import hashlib
import ipaddress
import json
import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARKERS = ROOT / ".github" / "private-markers.txt"
MANIFEST = ROOT / "docs" / "receipts-redaction.sha256"
ORIGIN = "069a029befd42b617e3da1682e76070831bc381f"   # the commit whose bytes are the git-held originals

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
GUARD_FILES = {".github/private-markers.txt", ".github/workflows/private-marker-guard.yml", "scripts/check_public_ids.py",
               "tests/test_receipts_redaction.py"}

# ---- an independent matcher: shape and context, never the transform's value list ----
HEX = rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
ZERO = rb"0{8}-0{4}-0{4}-0{4}-0{12}"   # the guard's canary
SHAPES = {
    "GPU UUID or its prefix": re.compile(rb"(?i)(?<![0-9a-z])(?:GPU|MIG)-(?!" + ZERO + rb")[0-9a-f]{6}"),
    "UUID after a uuid key": re.compile(rb"(?i)uuid[\"']?\s*[:=]\s*[\"']?(?!" + ZERO + rb")" + HEX),
    "container label": re.compile(rb"(?<![0-9A-Za-z])C\.[0-9]{7,10}(?![0-9])"),
    "provider ssh endpoint": re.compile(rb"(?i)ssh[0-9]+\.vast\.ai[: ]+(?:-p\s*)?[0-9]{2,5}|-p\s*[0-9]{2,5}\s+\S*@ssh[0-9]+\.vast\.ai"),
}
# a public IPv4 after "@", after an address word, or in a box token's parentheses; global only (loopback, private,
# CGNAT and documentation ranges pass), and never a dotted version: those follow "==", "-", "version" or a name
ADDRESS = re.compile(rb"(?i)(?:@|(?<![a-z_])(?:public_ipaddr|ipaddr|hostip|host|ip|addr|address)(?![a-z])[^0-9\n]{0,24}?"
                     rb"|box \*\*redacted-[0-9a-f]{8}\*\* \()((?:[0-9]{1,3}\.){3}[0-9]{1,3})(?![0-9.-])")


def is_global(ip):
    try:
        a = ipaddress.ip_address(ip.decode())
    except ValueError:
        return False
    return a.is_global and a not in ipaddress.ip_network("100.64.0.0/10")


PREFIX = b"GPU-" + b"1a2b3c4d"    # split, so no added line here carries the shapes these tests plant
HOST = b"ssh7." + b"vast.ai"
NUMBER = re.compile(rb"(?<![0-9A-Za-z.])[0-9]{7,10}(?![0-9A-Za-z]|\.[0-9])")
# a number 7-10 digits long, within 24 bytes after one of these words (a line break included), is an instance id;
# offer, ask and machine ids are a different class (the machine-id follow-up), so the nearest word decides
WORD = re.compile(rb"(?i)(?<![a-z])(instances?|box|vast|contract|container|list_after|offer|ask|machine)(?![a-z])")
NOT_INSTANCE = (b"offer", b"ask", b"machine")


def scan(data):
    """{rule: count} of every identifier shape in these bytes."""
    found = collections.Counter()
    for why, rx in SHAPES.items():
        found[why] += len(rx.findall(data))
    found["public IPv4 in an address context"] += sum(is_global(m.group(1)) for m in ADDRESS.finditer(data))
    for m in NUMBER.finditer(data):
        before = data[max(0, m.start() - 24):m.start()]
        if before.endswith(b"redacted-") or re.fullmatch(rb"20[0-9]{6}", m.group(0)):
            continue
        words = WORD.findall(before)
        if words and words[-1].lower() not in NOT_INSTANCE:
            found["instance id in context"] += 1
    return +found


def scan_file(path):
    data = path.read_bytes()
    found = scan(data)
    if path.name.endswith(".gz"):
        try:
            found += scan(gzip.decompress(data))
        except OSError:
            pass
    return found


ALLOWED = {   # (path, rule) -> (count, why it stays)
    ("tests/fixtures/dq11-census-refused/receipt.json", "instance id in context"):
        (1, "a Vast OFFER id inside Vast's own error text ('Instance type by id <offer>'); offer ids go to the "
            "machine-id follow-up with the fixture's offer_ids"),
    ("tests/test_host_facts.py", "instance id in context"):
        (2, "CPU clocks in kHz (max_khz) for a fixture directory named box, not an id"),
    ("tests/test_p159.py", "instance id in context"):
        (1, "a page-fault count constant (BOX.FAULTS_...), not an id"),
    ("tests/test_dq11_census_transport.py", "public IPv4 in an address context"):
        (10, "synthetic public_ipaddr, HostIp and ssh values the test feeds to the code that strips them from logs"),
}


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
    carry = {fx for fx in HOLDS if scan_file(ROOT / fx)["GPU UUID or its prefix"] + scan_file(ROOT / fx)["UUID after a uuid key"]}
    exempt = _exempt()
    assert set(exempt) == carry | LEDGER_FIXTURES
    for fx in carry:
        assert exempt[fx][1] == HOLDS[fx][1], f"{fx}: the guard's pin is not the hold's"


def test_no_identifier_outside_the_holds():
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    found = {}
    for fx in tracked.decode("utf-8").split("\0"):
        if not fx or fx in HOLDS or fx in TEST_SOURCES or fx in GUARD_FILES or fx in LEDGER_FIXTURES:
            continue
        try:
            hits = scan_file(ROOT / fx)
        except OSError:
            continue
        for why, n in hits.items():
            found[(fx, why)] = n
    allowed = {k: n for k, (n, _) in ALLOWED.items()}
    extra = {k: n for k, n in found.items() if allowed.get(k) != n}
    stale = {k: n for k, n in allowed.items() if found.get(k) != n}
    assert not extra, f"identifiers outside the holds (redact them, or rule an allowance): {dict(list(extra.items())[:5])}"
    assert not stale, f"allowances that no longer match the tree; remove or re-count them: {stale}"


@pytest.mark.parametrize("why,data", [
    ("GPU UUID or its prefix", b"\0bin\0GPU-" + b"-".join((b"1" * 8, b"2" * 4, b"3" * 4, b"4" * 4, b"5" * 12))),
    ("GPU UUID or its prefix", b"the card (" + PREFIX + b") was pinned"),
    ("UUID after a uuid key", b'{"uuid": "' + b"-".join((b"6" * 8, b"7" * 4, b"8" * 4, b"9" * 4, b"a" * 12)) + b'"}'),
    ("instance id in context", b'"vast_instance_id": ' + str(7 * 10 ** 7 + 4321).encode()),
    ("instance id in context", b'{"proof": "{\\"instance_id\\": \\"' + str(7 * 10 ** 7 + 4321).encode() + b'\\"}"}'),
    ("instance id in context", b"rented (Vast box " + str(7 * 10 ** 7 + 4321).encode() + b")"),
    ("instance id in context", b"| box **" + str(7 * 10 ** 7 + 4321).encode() + b"** |"),
    ("instance id in context", b"destroyed with Vast instance\n" + str(7 * 10 ** 7 + 4321).encode() + b" and gone"),
    ("container label", b"VAST_CONTAINERLABEL=C." + str(7 * 10 ** 7 + 4321).encode()),
    ("provider ssh endpoint", b"ssh root@" + HOST + b":" + str(4321).encode()),
    ("provider ssh endpoint", b"ssh -p " + str(4321).encode() + b" root@" + HOST),
    ("public IPv4 in an address context", b'{"public_ipaddr": "' + b".".join((b"81", b"2", b"3", b"4")) + b'"}'),
    ("public IPv4 in an address context", b"ssh root@" + b".".join((b"81", b"2", b"3", b"4"))),
    ("public IPv4 in an address context", b"box **redacted-0a0a0a0a** (" + b".".join((b"81", b"2", b"3", b"4")) + b", 56 MB/s)"),
    ("public IPv4 in an address context", b'"HostIp": "' + b".".join((b"81", b"2", b"3", b"4")) + b'"'),
])
def test_the_matcher_refuses_each_planted_identifier(why, data):
    assert scan(data)[why] >= 1, (why, data)


def test_the_matcher_leaves_tokens_canaries_and_other_numbers_alone():
    canary = b"GPU-" + b"-".join((b"0" * 8, b"0" * 4, b"0" * 4, b"0" * 4, b"0" * 12))
    ok = (b"GPU-redacted-1a2b3c4d instance redacted-12345678 box redacted-1a2b3c4d " + canary
          + b" GPU-accelerated; offer 70004321 on machine 12345; host_bytes 1234567890; box at 20261025; 1.23456789"
          + b" root@172.16.4.5 host 127.0.0.1 ip 203.0.113.9 nvidia-cublas==12.8.4.1 pkg@12.8.4.1-py3")
    assert scan(ok) == collections.Counter()
    assert scan(gzip.compress(PREFIX)) == collections.Counter()                      # bytes on disk, compressed
    assert scan(gzip.decompress(gzip.compress(PREFIX)))["GPU UUID or its prefix"] == 1


# ---- the manifest: every redacted file is its recorded bytes, every removed file is held by the store ----
def test_the_manifest_matches_the_tree():
    rows = [ln.split("  ", 3) for ln in MANIFEST.read_text(encoding="utf-8").splitlines() if ln and ln[0] != "#"]
    assert len(rows) == 2227
    removed = [r for r in rows if r[1] == "removed"]
    assert len(removed) == 7
    for orig, red, origin, fx in rows:
        assert origin in ("store", "git:" + ORIGIN[:8]), origin
        assert orig != red and fx not in HOLDS and fx not in TEST_SOURCES
        if red == "removed":
            assert origin == "store" and not (ROOT / fx).exists(), fx
            note = (ROOT / fx).parent / "STORE-ONLY.sha256"
            assert f"{orig}  {(ROOT / fx).name}" in note.read_text(encoding="utf-8"), f"{fx}: no STORE-ONLY line"
        else:
            assert hashlib.sha256((ROOT / fx).read_bytes()).hexdigest() == red, f"{fx} is not its redacted bytes"


def test_a_redacted_receipt_is_its_original_with_tokens_in_place():
    """Spot-check a git-held original: the redacted file is the original with identifier-shaped spans swapped for
    tokens, and nothing else; equal values take equal tokens."""
    rows = [ln.split("  ", 3) for ln in MANIFEST.read_text(encoding="utf-8").splitlines() if ln and ln[0] != "#"]
    checked = 0
    for orig, red, origin, fx in rows:
        if origin == "store" or red == "removed" or fx.endswith(".gz"):
            continue
        old = subprocess.run(["git", "-C", str(ROOT), "show", f"{ORIGIN}:{fx}"], capture_output=True, check=True).stdout
        assert hashlib.sha256(old).hexdigest() == orig, f"{fx}: the original is not git's bytes at {ORIGIN[:8]}"
        checked += 1
    assert checked > 300


# ---- the claims register: a redaction of a host descriptor, never a change of claim ----
# "Claims are never edited, only superseded" protects a claim's substance: its number, meaning, unit, status, tier and
# evidence. The redaction touched only the descriptive text fields below, and only by swapping identifiers for tokens.
TEXT_FIELDS = ("hardware", "notes", "claim")
TOKEN = re.compile(r"(?:GPU-|instance-|ssh-)?redacted-[0-9a-f]{8}")
SLOT = {"ssh-": r"(ssh[0-9]+\.vast\.ai:[0-9]{2,5})", "GPU-": r"(GPU-[0-9a-f-]{8,36})"}


def claims_problems(old, new):
    """Every way the branch's register differs from the originals other than identifiers turned into tokens."""
    out = []
    o, n = old["claims"], new["claims"]
    if [c["id"] for c in o] != [c["id"] for c in n]:
        return ["the claim ids or their order changed"]
    if {k: v for k, v in old.items() if k != "claims"} != {k: v for k, v in new.items() if k != "claims"}:
        out.append("a top-level field changed")
    pairs = {}
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
            # the NEW text, each token turned into a slot for the identifier shape it replaces, must match the OLD text
            parts, last, toks = [], 0, []
            for m in TOKEN.finditer(y):
                kind = next((p for p in SLOT if m.group(0).startswith(p)), "")
                parts += [re.escape(y[last:m.start()]), SLOT.get(kind, r"([0-9]{6,10})")]
                toks.append(m.group(0))
                last = m.end()
            parts.append(re.escape(y[last:]))
            got = re.fullmatch("".join(parts), x)
            if not toks or not got:
                out.append(f"{a['id']}: {k} differs by more than identifiers turned into tokens")
                continue
            for value, tok in zip(got.groups(), toks):
                key = tok.rsplit("-", 1)[1]
                if pairs.setdefault(value, key) != key:
                    out.append(f"{a['id']}: one identifier took two tokens")
    if len(set(pairs.values())) != len(pairs):
        out.append("two identifiers share one token")
    return out


def _registers():
    old = json.loads(subprocess.run(["git", "-C", str(ROOT), "show", f"{ORIGIN}:docs/claims.json"],
                                    capture_output=True, check=True).stdout)
    return old, json.loads((ROOT / "docs" / "claims.json").read_text(encoding="utf-8"))


def test_the_register_changed_only_by_identifiers_turned_into_tokens():
    old, new = _registers()
    assert claims_problems(old, new) == []
    assert sum(a != b for a, b in zip(old["claims"], new["claims"])) == 390            # the rows the redaction touched


def test_a_changed_digit_or_word_fails_the_register_check():
    old, new = _registers()
    bad = copy.deepcopy(new)
    row = next(c for c in bad["claims"] if isinstance(c.get("value"), (int, float)))
    row["value"] = row["value"] + 1
    assert any("value changed" in p for p in claims_problems(old, bad))
    bad = copy.deepcopy(new)
    row = next(c for c in bad["claims"] if "redacted-" in c.get("hardware", ""))
    row["hardware"] = row["hardware"].replace("RTX", "GTX", 1)
    assert any("more than identifiers turned into tokens" in p for p in claims_problems(old, bad))
    bad = copy.deepcopy(new)
    rows = [c for c in bad["claims"] if re.search(r"(?<![a-z-])redacted-[0-9a-f]{8}", c.get("hardware", ""))]
    tok = re.search(r"(?<![a-z-])redacted-([0-9a-f]{8})", rows[0]["hardware"]).group(1)
    other = next(re.search(r"(?<![a-z-])redacted-([0-9a-f]{8})", c["hardware"]).group(1) for c in rows
                 if tok not in c["hardware"])
    rows[0]["hardware"] = rows[0]["hardware"].replace(tok, other)                      # one value, someone else's token
    assert claims_problems(old, bad)


# ---- every ID_FILE (CHANGELOG.md's released sections among them) changed only by identifiers turned into tokens ----
FILE_TOKEN = re.compile(r"(box|instances?)-redacted-[0-9a-f]{8}|(?:GPU-|instance-|ssh-|ip-)?redacted-[0-9a-f]{8}")


def substitution_problems(old, new):
    """[] when NEW is OLD with identifier-shaped spans swapped for tokens, one value always taking one token."""
    parts, last, toks = [], 0, []
    for m in FILE_TOKEN.finditer(new):
        t = m.group(0)
        if m.group(1):   # "box<digits>" became "box-redacted-<8 hex>"; "instance-redacted-" also stands for a bare id
            slot = "(?:" + re.escape(m.group(1)) + r")?([0-9]{6,10})"
        elif t.startswith("ssh-"):
            slot = r"(ssh[0-9]+\.vast\.ai:[0-9]{2,5})"
        elif t.startswith("ip-"):
            slot = r"((?:[0-9]{1,3}\.){3}[0-9]{1,3})"
        elif t.startswith("GPU-"):
            slot = r"GPU-([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}|[0-9a-fA-F]{8})"
        else:
            slot = r"([0-9]{6,10})"
        parts += [re.escape(new[last:m.start()]), slot]
        toks.append(t.rsplit("-", 1)[1])
        last = m.end()
    parts.append(re.escape(new[last:]))
    got = re.fullmatch("".join(parts), old)
    if not got:
        return ["differs by more than identifiers turned into tokens"]
    pairs, out = {}, []
    for value, tok in zip(got.groups(), toks):
        v = value.lower()[:8] if not v_is_digits(value) else value     # a UUID and its prefix share one token
        if pairs.setdefault(v, tok) != tok:
            out.append("one identifier took two tokens")
    if len(set(pairs.values())) != len(pairs):
        out.append("two identifiers share one token")
    return out


def v_is_digits(value):
    return value.isdigit() or value.startswith("ssh") or value.count(".") == 3


def _id_files():
    import importlib.util
    spec = importlib.util.spec_from_file_location("redact_receipt_ids", ROOT / "scripts" / "redact_receipt_ids.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return sorted(mod.ID_FILES)


@pytest.mark.parametrize("fx", _id_files())
def test_an_id_file_changed_only_by_identifiers_turned_into_tokens(fx):
    old = subprocess.run(["git", "-C", str(ROOT), "show", f"{ORIGIN}:{fx}"], capture_output=True, check=True).stdout
    new = (ROOT / fx).read_bytes()
    assert old != new, f"{fx} is an ID_FILE the redaction did not touch"
    assert substitution_problems(old.decode("utf-8"), new.decode("utf-8")) == [], fx


def test_the_file_substitution_check_refuses_a_changed_word_or_a_reused_token():
    inst, other = str(7 * 10 ** 7 + 4321), str(7 * 10 ** 7 + 4322)
    old = f"box {inst} ran; box {other} ran; instance{inst} gone"
    good = "box redacted-0a0a0a0a ran; box redacted-0b0b0b0b ran; instance-redacted-0a0a0a0a gone"
    assert substitution_problems(old, good) == []
    assert substitution_problems(old, good.replace("ran;", "failed;", 1))                    # a word changed
    assert substitution_problems(old, good.replace("0b0b0b0b", "0a0a0a0a"))                 # two values, one token
    assert substitution_problems(old, good.replace("instance-redacted-0a0a0a0a", "instance-redacted-0c0c0c0c"))


# ---- the transform's own contract, on synthetic values ----
def _transform():
    import importlib.util
    spec = importlib.util.spec_from_file_location("redact_receipt_ids", ROOT / "scripts" / "redact_receipt_ids.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tokens_keep_identity_and_lose_the_shape():
    R = _transform()
    one = "-".join(("1" * 8, "2" * 4, "3" * 4, "4" * 4, "5" * 12))   # built here so this file names none
    two = "-".join(("6" * 8, "7" * 4, "8" * 4, "9" * 4, "a" * 12))
    inst = str(7 * 10 ** 7 + 4321)
    red = R.Redactor({"gpu": {one: "0a0a0a0a", two: "0b0b0b0b"}, "instance": {inst: "0c0c0c0c"}})
    text = red.redact("bench/x/RESULTS.md", f'GPU-{one} GPU-{one} GPU-{two} GPU-{one[:8]} "vast_instance_id": {inst} '
                                            f"Vast instance {inst} box{inst}")
    assert scan(text.encode()) == collections.Counter()
    assert text.count("GPU-redacted-0a0a0a0a") == 3 and "GPU-redacted-0b0b0b0b" in text   # the prefix takes its UUID's token
    assert '"vast_instance_id": "instance-redacted-0c0c0c0c"' in text                    # a JSON number becomes a string
    assert "Vast instance redacted-0c0c0c0c" in text and "box-redacted-0c0c0c0c" in text
    alone = f"a count of {inst} rows"
    assert red.redact("bench/x/RESULTS.md", alone) == alone                            # no id word, no ID_FILES: untouched
    zero = "GPU-" + "-".join(("0" * 8, "0" * 4, "0" * 4, "0" * 4, "0" * 12))
    assert red.redact("x.md", zero) == zero                                            # the guard's canary stays
