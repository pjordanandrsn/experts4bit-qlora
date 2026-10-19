"""HP1's one evaluator (bench/hp1/hp1_eval.py), its rules with the model mocked: every STRICT failure VOIDs, fail-closed, and a
TC1-invalid run is never scored. Each test injects one failure into an otherwise passing pair of arms."""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "hp1"))
import hp1_eval  # noqa: E402

TOK = "tokens-sha"
DIG = "d" * 64


def arm(tag, **kw):
    a = {"receipt": f"{tag}.json", "framework": "e4b", "tag": tag, "seed": 3410, "order": {"order_seed": 3410, "order_sha": "o"},
         "tokens_sha": TOK, "adapter": f"{tag}.safetensors", "adapter_sha256": f"sha-{tag}", "train_digest": DIG,
         "inloop_step0": 2.0000, "inloop_final": 1.5000, "inloop_final_step": 200, "steps": 200, "hp1_errors": {}, "raw_ok": True,
         "status_receipt": "ok", "c1_bit_exact": True, "trainable_params": 642514944, "lora_init": "matched:3410",
         "adapter_dtype": "native", "matched_init": {"complete": True}, "adapter_dtypes_after": {"torch.float32": 16, "torch.bfloat16": 8}}
    a.update(kw)
    return a


class Store:
    def __init__(self, digest=DIG, want=DIG):
        self._d, self.meta = digest, {"train_base_digest": want}

    def digest(self):
        return self._d, 533


class Harness:
    """The injected scorer: zero adapter -> `zero`; an adapter -> its `target` row mean (by tag); a perturbed adapter -> `flip`."""

    def __init__(self, zero=2.0010, targets=None, flip_moves=True, stores=None, zero_rows=None):
        self.zero, self.targets, self.flip_moves, self.zero_rows = zero, targets or {}, flip_moves, zero_rows
        self.stores = stores or {"base_e4b": Store()}
        self.scored, self.cur = [], None

    def open_base(self, d):
        return self.stores[d]

    def make_scorer(self, store):
        def set_adapter(canon):
            self.cur = canon

        def score():
            if self.cur is None:
                return {"row_mean": self.zero, "token_weighted": self.zero, "row_losses": self.zero_rows}
            t, _ = self.cur
            tag = t["tag"]
            v = self.targets.get(tag, 1.5010)
            if float(t["0.q.B"].view(-1)[0]) != 0.0 and self.flip_moves:
                v += 0.25
            self.scored.append(tag)
            return {"row_mean": v, "token_weighted": v}
        return set_adapter, score

    def load_adapter(self, path):
        return {"tag": path.split(".")[0], "0.q.B": torch.zeros(2, 2)}, {"0.q": 1.0}

    def run(self, arms, bases=("base_e4b",), expect=642514944, **kw):
        return hp1_eval.evaluate(copy.deepcopy(arms), list(bases), rows=None, open_base=self.open_base, make_scorer=self.make_scorer,
                                 load_adapter=self.load_adapter, file_sha=lambda p: "sha-" + p.split(".")[0], tokens_sha=TOK,
                                 expect_trainable=expect, **kw)


def by_tag(arms):
    return {a["tag"]: a for a in arms}


def test_the_passing_pair_is_scored():
    h = Harness()
    bases, arms = h.run([arm("a"), arm("b")])
    assert bases[0]["status"] == "OK" and bases[0]["perturbed_adapter_control"]["changed"]
    assert {a["status"] for a in arms} == {"SCORED"}
    assert by_tag(arms)["a"]["evaluator"]["row_mean"] == 1.5010


def test_a_base_whose_digest_fails_its_training_run_voids_its_arms():
    h = Harness(stores={"base_e4b": Store(digest="x" * 64, want=DIG)})
    bases, arms = h.run([arm("a"), arm("b")])
    assert bases[0]["status"] == "VOID"
    assert {a["status"] for a in arms} == {"VOID"} and "digest" in arms[0]["reason"] and not h.scored


def test_the_zero_adapter_check_voids_the_base_and_its_arms():
    h = Harness(zero=2.0060)                       # 0.006 > 0.005 from the arms' step 0
    bases, arms = h.run([arm("a"), arm("b")])
    assert bases[0]["status"] == "VOID" and "zero-adapter" in bases[0]["reason"]
    assert {a["status"] for a in arms} == {"VOID"} and not h.scored


@pytest.mark.parametrize("field,value,needle", [
    ("inloop_step0", None, "step-0"),
    ("status_receipt", "c1_failed", "receipt status"),
    ("c1_bit_exact", False, "C1"),
    ("trainable_params", 642514943, "trainable"),
    ("matched_init", {"complete": False}, "matched init"),
    ("adapter_dtypes_after", {"torch.float32": 24}, "dtype census"),
    ("order", None, "order"),
    ("tokens_sha", "other", "tokens sha"),
    ("raw_ok", False, "raw_ok"),
    ("inloop_final_step", 199, "step-200"),
])
def test_a_tc1_invalid_run_is_never_scored(field, value, needle):
    h = Harness()
    bases, arms = h.run([arm("a", **{field: value}), arm("b")])
    a = by_tag(arms)["a"]
    assert a["status"] == "VOID" and needle in a["reason"], a.get("reason")
    assert "a" not in h.scored and by_tag(arms)["b"]["status"] == "SCORED"


def test_an_fp32_cell_with_a_bf16_adapter_is_invalid():
    h = Harness()
    _, arms = h.run([arm("a", adapter_dtype="fp32"), arm("b")])
    assert by_tag(arms)["a"]["status"] == "VOID" and "fp32" in by_tag(arms)["a"]["reason"]


def test_the_round_trip_check_voids_a_mis_exported_adapter():
    h = Harness(targets={"a": 1.5300})             # 0.03 off its in-loop step-200 score
    _, arms = h.run([arm("a"), arm("b")])
    assert by_tag(arms)["a"]["status"] == "VOID" and "round-trip" in by_tag(arms)["a"]["reason"]
    assert by_tag(arms)["b"]["status"] == "SCORED"


def test_a_dead_perturbation_control_voids_the_base_and_its_arms():
    h = Harness(flip_moves=False)
    bases, arms = h.run([arm("a"), arm("b")])
    assert bases[0]["status"] == "VOID" and {a["status"] for a in arms} == {"VOID"}


def test_an_arm_no_export_claims_is_a_run_level_void():
    """Every verdict uses evaluator numbers only: an arm whose training digest no export claims has none, so it is VOID (run-level,
    redrawn once with its base re-exported), never scored on its in-loop number."""
    h = Harness()
    _, arms = h.run([arm("a", train_digest="e" * 64), arm("b")])
    a = by_tag(arms)["a"]
    assert a["status"] == "VOID" and a["void_level"] == "run" and "no base export" in a["reason"]
    assert by_tag(arms)["b"]["status"] == "SCORED"


def test_void_levels_split_run_from_base():
    h = Harness(zero=2.0060)
    _, arms = h.run([arm("a"), arm("b", status_receipt="c1_failed")])
    assert by_tag(arms)["a"]["void_level"] == "base" and by_tag(arms)["b"]["void_level"] == "run"


def test_a_duplicate_base_is_unused_and_does_not_void():
    h = Harness(stores={"base_e4b": Store(), "base_unsloth": Store()})
    bases, arms = h.run([arm("a"), arm("b", framework="unsloth", adapter_dtype="native", lora_init="native",
                                        adapter_dtypes_after={"torch.float32": 24})], bases=("base_e4b", "base_unsloth"))
    assert [b["status"] for b in bases] == ["OK", "UNUSED"] and {a["status"] for a in arms} == {"SCORED"}


def test_each_base_is_released_before_the_next_loads():
    """Regression (A2000 shakedown run 3): the scorer closes over the model, so evaluate() drops it and calls release() after
    every base -- one model on the device at a time (a 12 GB card held two and ran out of memory on the duplicate base)."""
    import weakref
    h = Harness(stores={"base_e4b": Store(), "base_unsloth": Store()})
    live, calls = [], []

    class Model:
        pass

    base_make = h.make_scorer

    def make_scorer(store):
        m = Model()
        live.append(weakref.ref(m))
        sa, sc = base_make(store)
        return (lambda c, m=m: sa(c)), (lambda m=m: sc())

    def release():
        import gc
        gc.collect()
        calls.append(sum(r() is not None for r in live))
    hp1_eval.evaluate([arm("a")], ["base_e4b", "base_unsloth"], None, h.open_base, make_scorer, h.load_adapter,
                      lambda p: "sha-" + p.split(".")[0], TOK, 642514944, release=release)
    assert calls == [0, 0], calls


def test_an_arm_on_its_own_base_is_scored_by_the_evaluator_on_that_base():
    """(a) An arm whose training base differs from the shared one is scored by the EVALUATOR on its OWN exported base (matched by
    digest, the same STRICT checks on that base) -- usable, with an evaluator number, never an in-loop one."""
    other = "f" * 64
    h = Harness(stores={"base_e4b": Store(), "base_axolotl": Store(digest=other, want=other)})
    bases, arms = h.run([arm("a"), arm("x", framework="axolotl", lora_init="native", adapter_dtypes_after={"torch.float32": 24},
                                        train_digest=other)], bases=("base_e4b", "base_axolotl"))
    assert [b["status"] for b in bases] == ["OK", "OK"]
    x = by_tag(arms)["x"]
    assert x["status"] == "SCORED" and x["base"] == "base_axolotl" and x["evaluator"]["row_mean"] == 1.5010


def test_the_zero_adapter_tolerance_is_a_parameter_and_is_recorded():
    """HP1-PREREG Amendment 4: the 48-layer tolerance comes from the calibration; the default stays the 2-layer 0.005."""
    bases, _ = Harness(zero=2.0060).run([arm("a")])
    assert bases[0]["status"] == "VOID" and bases[0]["zero_adapter_tol"] == hp1_eval.ZERO_ADAPTER_TOL == 0.005
    bases, arms = Harness(zero=2.0060).run([arm("a")], zero_tol=0.01)
    assert bases[0]["status"] == "OK" and bases[0]["zero_adapter_tol"] == 0.01 and arms[0]["status"] == "SCORED"
    bases, _ = Harness(zero=2.0060).run([arm("a")], zero_tol=0.0059)
    assert bases[0]["status"] == "VOID" and "> 0.0059" in bases[0]["reason"]


def test_per_row_zero_adapter_losses_and_gaps_are_recorded():
    """Amendment 4 item 5: per-row zero-adapter losses, and per-row gaps against the arm's in-loop step-0 rows, so a VOID can be
    diagnosed from the receipt. Gaps only where the receipt carries step-0 rows of the same length."""
    h = Harness(zero=2.0060, zero_rows=[2.0, 2.012])
    bases, _ = h.run([arm("a", inloop_step0_rows=[1.999, 2.001]), arm("b", inloop_step0_rows=[2.0]), arm("c")])
    b = bases[0]
    assert b["zero_adapter"]["row_losses"] == [2.0, 2.012] and b["status"] == "VOID"      # recorded on a VOID too
    assert b["zero_adapter_row_gaps"] == {"a": [0.001, 0.011]}                            # b: length mismatch; c: no rows


def test_hp2_zero_adapter_report_only_records_but_never_voids():
    """HP2: the zero-adapter gap is deterministic per base (a rerun cannot change it), so it is a reported fact, not a VOID."""
    bases, arms = Harness(zero=2.0300).run([arm("a")], zero_report_only=True)
    assert bases[0]["status"] == "OK" and bases[0]["zero_adapter_outside_tol"] == ["a"] and bases[0]["zero_adapter_report_only"]
    assert arms[0]["status"] == "SCORED"


def test_hp2_round_trip_is_centred_on_each_runs_own_zero_adapter_gap():
    """HP2 (#1793 ruling): 6(b) VOIDs iff |(evaluator - in-loop N) - (zero adapter - in-loop 0)| > tol, so it tests only the adapter's
    round trip; the zero-adapter gap itself (deterministic per base) is reported. The 0.02 constant does not also apply."""
    h = Harness(zero=2.0250, targets={"a": 1.5300, "b": 1.5400})              # z = +0.025; a: 0.030 - 0.025 = 0.005; b: 0.015
    bases, arms = h.run([arm("a"), arm("b")], zero_report_only=True, roundtrip_zero_centred=0.01)
    got = by_tag(arms)
    assert got["a"]["status"] == "SCORED" and got["a"]["zero_gap"] == 0.025 and got["a"]["adapter_roundtrip"] == 0.005
    assert got["b"]["status"] == "VOID" and "zero-centred" in got["b"]["reason"] and got["b"]["void_level"] == "run"
    assert got["b"]["adapter_roundtrip"] == 0.015 and bases[0]["status"] == "OK"
