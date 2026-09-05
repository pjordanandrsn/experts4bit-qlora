"""scripts/check_capabilities.py: the per-path ``training_support`` cross-reference.

A capability's training support is structured, never a flat flag: six paths per
model_type, each with a status from a fixed vocabulary; ``supported`` / ``void`` /
``refused`` cite claim ids that exist; ``model_families`` equals the model_types
whose ``fast_train`` is ``supported``. These tests pin that contract on a fixture
so a register edit cannot silently promote a family.
"""
import copy
import importlib.util
import pathlib

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_capabilities.py"
_spec = importlib.util.spec_from_file_location("check_capabilities", _SCRIPT)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

BY_ID = {
    "c.pass": {"id": "c.pass", "status": "measured"},
    "c.void": {"id": "c.void", "status": "measured"},
    "c.open": {"id": "c.open", "status": "open"},
}


def _cap():
    paths = {p: {"status": "not_tested"} for p in cc.TRAINING_PATHS}
    good = copy.deepcopy(paths)
    good.update({
        "quantize": {"status": "supported", "claim_ids": ["c.pass"]},
        "reference_train": {"status": "supported", "claim_ids": ["c.pass"]},
        "fast_train": {"status": "supported", "claim_ids": ["c.pass"]},
        "batched_train": {"status": "void", "claim_ids": ["c.void"], "reason": "kernel not reached on every layer"},
        "native_mxfp4_train": {"status": "n/a"},
    })
    bare = copy.deepcopy(paths)
    bare.update({
        "quantize": {"status": "supported", "claim_ids": ["c.pass"]},
        "fast_train": {"status": "refused", "claim_ids": ["c.pass"], "reason": "experts built bare"},
        "batched_train": {"status": "refused", "claim_ids": ["c.pass"], "reason": "experts built bare"},
        "native_mxfp4_train": {"status": "experimental"},
    })
    return {"id": "cap", "model_families": ["good_moe"],
            "training_support": {"headline_path": "fast_train", "by_model_type": {"good_moe": good, "bare_moe": bare}}}


def _paths(cap):
    return cap["training_support"]["by_model_type"]


def _errors(cap):
    errors = []
    cc.check_training_support(cap, cap["id"], BY_ID, errors)
    return errors


def test_valid_object_passes():
    assert _errors(_cap()) == []


def test_absent_object_is_not_an_error():
    cap = _cap()
    del cap["training_support"]
    assert _errors(cap) == []


def test_unknown_status_is_rejected():
    cap = _cap()
    _paths(cap)["good_moe"]["nvme_train"]["status"] = "works"
    assert any("not in" in e and "nvme_train" in e for e in _errors(cap))


def test_supported_must_cite_an_existing_active_claim():
    cap = _cap()
    _paths(cap)["good_moe"]["quantize"]["claim_ids"] = []
    assert any("must cite at least one claim id" in e for e in _errors(cap))
    cap = _cap()
    _paths(cap)["good_moe"]["quantize"]["claim_ids"] = ["c.missing"]
    assert any("claim id not in docs/claims.json: c.missing" in e for e in _errors(cap))
    cap = _cap()
    _paths(cap)["good_moe"]["quantize"]["claim_ids"] = ["c.open"]
    assert any("only" in e and "can back a supported path" in e for e in _errors(cap))


def test_void_and_refused_cite_a_claim_and_refused_says_why():
    cap = _cap()
    _paths(cap)["good_moe"]["batched_train"]["claim_ids"] = []
    assert any("batched_train" in e and "must cite" in e for e in _errors(cap))
    cap = _cap()
    _paths(cap)["bare_moe"]["fast_train"].pop("reason")
    assert any("'refused' must carry a reason" in e for e in _errors(cap))


def test_model_families_equals_fast_train_supported():
    cap = _cap()
    cap["model_families"] = ["good_moe", "bare_moe"]          # a flat promotion with no supported fast_train
    assert any("model_families" in e and "fast_train" in e for e in _errors(cap))
    cap = _cap()
    cap["model_families"] = []                                # a supported family left off the list
    assert any("model_families" in e for e in _errors(cap))


def test_every_path_is_required():
    cap = _cap()
    del _paths(cap)["good_moe"]["nvme_train"]
    assert any("nvme_train: missing status" in e for e in _errors(cap))


def test_null_headline_skips_the_model_families_rule_but_not_the_rest():
    cap = _cap()
    cap["training_support"]["headline_path"] = None
    cap["model_families"] = ["anything", "at all"]                # the list is the capability's own scope
    assert _errors(cap) == []
    _paths(cap)["good_moe"]["quantize"]["claim_ids"] = ["c.missing"]
    assert any("c.missing" in e for e in _errors(cap))


def test_unknown_headline_path_is_rejected():
    cap = _cap()
    cap["training_support"]["headline_path"] = "vibes"
    assert any("headline_path" in e for e in _errors(cap))
