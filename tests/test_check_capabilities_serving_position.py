"""scripts/check_capabilities.py: a serving capability that cites none of the newest serving lane STATUS quotes.

The audit found ``serve-moe-on-consumer-gpu`` citing only 2026-09-04 rows while
docs/STATUS.md called the 2026-09-05 census the position -- every cited id
was ``measured``, which was all the checker asked. The rule is a WARN, never
a failure, and applies to capabilities whose primary mode is ``serving``.
"""
import importlib.util
import pathlib

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_capabilities.py"
_spec = importlib.util.spec_from_file_location("check_capabilities", _SCRIPT)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

BY_ID = {
    "e4b.serve.old": {"id": "e4b.serve.old", "area": "serve", "status": "measured", "measured_on": "2026-09-04"},
    "e4b.serve.new.b1": {"id": "e4b.serve.new.b1", "area": "serve", "status": "measured", "measured_on": "2026-09-05"},
    "e4b.serve.new.b16": {"id": "e4b.serve.new.b16", "area": "serve", "status": "measured", "measured_on": "2026-09-05"},
    "e4b.train.x": {"id": "e4b.train.x", "area": "train", "status": "measured", "measured_on": "2026-09-06"},
    "e4b.serve.gone": {"id": "e4b.serve.gone", "area": "serve", "status": "superseded", "measured_on": "2026-09-07"},
}
STATUS = "position: `e4b.serve.new.*` (2026-09-05); before it `e4b.serve.old`; training `e4b.train.x`; `e4b.serve.gone` superseded."


def test_newest_serving_lane_is_the_latest_dated_active_serve_claims_status_quotes():
    newest, ids = cc.newest_serving_lane(STATUS, BY_ID)
    assert newest == "2026-09-05"
    assert ids == {"e4b.serve.new.b1", "e4b.serve.new.b16"}       # not the train row, not the superseded one


def test_a_serving_capability_citing_only_the_older_lane_warns():
    doc = {"capabilities": [{"id": "serve", "modes": ["serving", "inference"], "claim_ids": ["e4b.serve.old"]}]}
    w = cc.serving_position_warnings(doc, BY_ID, STATUS)
    assert len(w) == 1 and "serve: primary mode 'serving'" in w[0] and "2026-09-05" in w[0]


def test_citing_any_row_of_the_newest_lane_or_not_being_primarily_serving_is_quiet():
    doc = {"capabilities": [{"id": "serve", "modes": ["serving"], "claim_ids": ["e4b.serve.old", "e4b.serve.new.b16"]},
                            {"id": "offload", "modes": ["offload", "serving"], "claim_ids": ["e4b.serve.old"]}]}
    assert cc.serving_position_warnings(doc, BY_ID, STATUS) == []
    assert cc.serving_position_warnings(doc, BY_ID, "no ids here") == []


def test_a_warning_is_also_a_github_actions_annotation():
    assert cc.warning_lines("serve: x") == ["WARN: serve: x", "::warning::serve: x"]


def test_the_id_namespace_is_the_registers_own():
    pat = cc._status_id_pattern(BY_ID)
    assert pat.findall("`e4b.serve.old` and `gnf4.kernel.x`") == ["e4b.serve.old"]
    gnf4 = {"gnf4.kernel.a": {}, "gnf4.serve.b": {}}
    assert cc._status_id_pattern(gnf4).findall("`e4b.serve.old` and `gnf4.serve.b`") == ["gnf4.serve.b"]


def test_a_register_without_one_namespace_quotes_nothing():
    mixed = dict(BY_ID, **{"gnf4.serve.z": {"id": "gnf4.serve.z", "area": "serve", "status": "measured",
                                            "measured_on": "2026-09-09"}})
    assert cc._status_id_pattern(mixed) is None
    assert cc.newest_serving_lane(STATUS, mixed) == (None, set())
    assert cc._status_id_pattern({}) is None


def test_the_serving_rule_runs_for_the_runtime_role_only(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    assert cc._system_role(root, "experts4bit-qlora") == "runtime"
    assert cc._system_role(root, "grouped-nf4-gemm") == "kernels"
    assert cc._system_role(root, "something-else") is None
    assert cc._system_role(tmp_path, "experts4bit-qlora") is None      # no manifest: no role, no rule


# The real STATUS layout: the position, then what changed, then what is open. On 2026-09-23 the rule read
# the whole file, so P61's diagnostic read -- quoted under "What is open", newer than every position row --
# became "the position" and the rule warned on every CI run.
SECTIONED = ("# Status\n\n## What you get today\n\nposition: `e4b.serve.new.*`; before it `e4b.serve.old`.\n\n"
             "## What changed — retired, superseded, corrected\n\n`e4b.serve.gone` superseded.\n\n"
             "## What is open\n\na newer diagnostic read: `e4b.serve.diag`.\n")
BY_ID_DIAG = {**BY_ID, "e4b.serve.diag": {"id": "e4b.serve.diag", "area": "serve", "status": "measured",
                                          "measured_on": "2026-09-23"}}
CITES_NEW = {"capabilities": [{"id": "serve", "modes": ["serving"], "claim_ids": ["e4b.serve.new.b1"]}]}


def test_a_lane_quoted_only_after_what_changed_is_not_the_position():
    assert cc.newest_serving_lane(SECTIONED, BY_ID_DIAG)[0] == "2026-09-23"      # the whole file: the diagnostic read
    assert cc.newest_serving_lane(cc.status_position_text(SECTIONED), BY_ID_DIAG)[0] == "2026-09-05"
    assert cc.serving_position_warnings(CITES_NEW, BY_ID_DIAG, SECTIONED) == []


def test_a_newer_lane_in_the_position_section_still_warns():
    status = SECTIONED.replace("position: `e4b.serve.new.*`", "position: `e4b.serve.diag`, before it `e4b.serve.new.*`")
    w = cc.serving_position_warnings(CITES_NEW, BY_ID_DIAG, status)
    assert len(w) == 1 and "2026-09-23" in w[0] and "e4b.serve.diag" in w[0]

