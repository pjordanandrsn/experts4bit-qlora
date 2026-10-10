"""CPU-only proposed guard checks; no model or GPU execution."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
LANE = ROOT / "bench/dq11"
sys.path.insert(0, str(LANE))
import dq11_a2000_quality as quality  # noqa: E402


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    for key in tuple(__import__("os").environ):
        if key.startswith("DQ11_REHEARSAL"):
            monkeypatch.delenv(key)
    for n in quality.PINS:
        shutil.copy2(LANE / ("rehearsal/wheels.json" if n == "wheels.json" else n), tmp_path / n)
    monkeypatch.setenv("E4B_SHA", "0" * 40)
    monkeypatch.setenv("TC1_RUN_NONCE", "cpu-fixture-no-gpu")
    marker = {
        "schema": quality.SCHEMA,
        "science_eligible": False,
        "optimizer_updates": 0,
        "source": "0" * 40,
        "nonce": "cpu-fixture-no-gpu",
        "pinned_files": copy.deepcopy(quality.PINS),
    }
    (tmp_path / "A2000-INITIAL-QUALITY.json").write_text(json.dumps(marker))
    return tmp_path, marker


def test_exact_real_marker_and_explicit_claim_guard(sealed, monkeypatch):
    root, marker = sealed
    assert quality.mode(root) == marker
    monkeypatch.delenv("DQ11_GPU_CLAIM_CONFIRMED", raising=False)
    with pytest.raises(ValueError, match="ownership"):
        quality.require_hardware(root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("science_eligible", True),
        ("optimizer_updates", 1),
        ("optimizer_updates", False),
        ("source", "1" * 40),
        ("nonce", "other"),
        ("schema", "dq11-rehearsal/1"),
        ("pinned_files", {}),
    ],
)
def test_bad_marker_cannot_enter_quality(sealed, field, value):
    root, marker = sealed
    marker[field] = value
    (root / "A2000-INITIAL-QUALITY.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError):
        quality.mode(root)


@pytest.mark.parametrize("name", list(quality.PINS))
def test_changed_real_manifest_refuses(sealed, name):
    root, _ = sealed
    with (root / name).open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="manifest changed"):
        quality.mode(root)


def test_tiny_mode_cannot_be_combined(sealed, monkeypatch):
    root, _ = sealed
    monkeypatch.setenv("DQ11_REHEARSAL", "1")
    with pytest.raises(ValueError, match="tiny rehearsal"):
        quality.mode(root)


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("padding_idx", [None, 0])
def test_loader_pad_zeroing_refuses_and_checkpoint_restore_matches_without_changing_L(sealed, sharded, padding_idx):
    import torch
    from safetensors.torch import save_file

    from dq11_arm import restore_padding_embedding
    from dq11_common import base_census, value_identity

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 4, padding_idx=padding_idx, dtype=torch.bfloat16)
            self.lm_head = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
            self.requires_grad_(False)

        def get_input_embeddings(self):
            return self.embed_tokens

    root, marker = sealed
    models = {arm: Tiny() for arm in ("L", "U", "U0")}
    original = torch.arange(1, 33, dtype=torch.bfloat16).reshape(8, 4)
    with torch.no_grad():
        models["L"].embed_tokens.weight.copy_(original)
    for arm in ("U", "U0"):
        models[arm].load_state_dict(models["L"].state_dict())
        with torch.no_grad():
            models[arm].embed_tokens.weight[0].zero_()
    snapshot = root / "cpu-tiny-checkpoint"
    snapshot.mkdir()
    name = "model-00001-of-00001.safetensors" if sharded else "model.safetensors"
    save_file({"model.embed_tokens.weight": original}, str(snapshot / name))
    if sharded:
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.embed_tokens.weight": name}})
        )
    rows = initial_rows_fixture(root, marker)
    for arm in rows:
        rows[arm]["base"] = base_census(models[arm])
    write_initial_rows(root, rows)
    with pytest.raises(ValueError, match="matched receipt identity"):
        quality.report(root)
    before_L = base_census(models["L"])
    for arm in ("U", "U0"):
        before_head = models[arm].lm_head.weight.detach().clone()
        restored = restore_padding_embedding(models[arm], snapshot)
        assert restored["row"] == 0
        assert torch.equal(models[arm].lm_head.weight, before_head)
        rows[arm]["base"] = base_census(models[arm])
        assert value_identity(rows[arm]["base"]) == value_identity(before_L)
    assert base_census(models["L"]) == before_L
    write_initial_rows(root, rows)
    assert quality.report(root)["status"] == "INITIAL_QUALITY_PASS"


@pytest.mark.parametrize("mutation", ["none", "name", "sm", "cuda", "torch", "count", "free", "total"])
def test_fake_hardware_requires_registered_identity_and_bounded_memory(sealed, monkeypatch, mutation):
    root, _ = sealed
    monkeypatch.setenv("DQ11_GPU_CLAIM_CONFIRMED", "1")
    attrs = {
        "name": "NVIDIA RTX A2000 12GB",
        "sm": (8, 6),
        "cuda": "12.8",
        "torch": "2.11.0+cu128",
        "count": 1,
        "free": 10 * (1 << 30),
        "total": 12 * (1 << 30),
    }
    if mutation != "none":
        attrs[mutation] = {
            "name": "NVIDIA GeForce RTX 5090",
            "sm": (12, 0),
            "cuda": "13.0",
            "torch": "2.12.1+cu130",
            "count": 2,
            "free": 8 * (1 << 30),
            "total": 32 * (1 << 30),
        }[mutation]
    fractions = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: attrs["count"],
        get_device_name=lambda: attrs["name"],
        get_device_capability=lambda: attrs["sm"],
        get_device_properties=lambda device: SimpleNamespace(total_memory=attrs["total"]),
        mem_get_info=lambda: (attrs["free"], attrs["total"]),
        set_per_process_memory_fraction=fractions.append,
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda=attrs["cuda"]), __version__=attrs["torch"]),
    )
    if mutation == "none":
        report = quality.require_hardware(root)
        assert report["science_eligible"] is False and report["optimizer_updates"] == 0
        assert report["process_cap_bytes"] == 9 * (1 << 30) and fractions == [0.75]
    else:
        with pytest.raises(ValueError):
            quality.require_hardware(root)
        assert fractions == []


def test_scientific_preparation_default_still_refuses_a2000_even_with_quality_marker(sealed, monkeypatch):
    # Execute the real proposed preparation entrypoint only to its GPU guard,
    # with external imports replaced by CPU stubs. No model construction.
    root, _ = sealed
    monkeypatch.syspath_prepend(str(LANE))
    s = importlib.util.spec_from_file_location("arm_proposal", LANE / "dq11_arm.py")
    arm = importlib.util.module_from_spec(s)
    s.loader.exec_module(arm)
    cuda = SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda: "NVIDIA RTX A2000 12GB"
    )
    flags = SimpleNamespace(allow_tf32=True)
    torch = SimpleNamespace(
        cuda=cuda,
        manual_seed=lambda seed: None,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=flags), cudnn=SimpleNamespace(allow_tf32=True)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "loggetta",
        SimpleNamespace(
            Constraints=lambda **kw: None,
            Workload=lambda **kw: None,
            plan=lambda *a, **kw: pytest.fail("science reached plan on A2000"),
        ),
    )
    monkeypatch.setitem(sys.modules, "loggetta.backends", SimpleNamespace(dense=object(), dense_train=object()))
    monkeypatch.setitem(sys.modules, "loggetta.hardware", SimpleNamespace(probe=lambda: None))
    with pytest.raises(ValueError, match="unregistered GPU"):
        arm.prepare(root, "L")
    with pytest.raises(ValueError, match="boolean"):
        arm.prepare(root, "L", quality_read=1)
    assert flags.allow_tf32 is False


def test_worker_refuses_no_claim_before_any_model_import(sealed, monkeypatch):
    root, _ = sealed
    monkeypatch.delenv("DQ11_GPU_CLAIM_CONFIRMED", raising=False)
    with pytest.raises(ValueError, match="before imports"):
        quality.worker(root, "L")


def initial_rows_fixture(root, marker):
    """Synthetic CPU receipts, not measurements of the registered model."""
    import hashlib
    import math

    from dq11_common import file_sha, object_sha

    (root / "receipts").mkdir()
    (root / "science.sha256").write_text("synthetic CPU seal\n")
    (root / "source_authority.json").write_text(json.dumps({"e4b_sha": marker["source"], "sha256": []}))
    versions = {p["name"]: p["version"] for p in json.loads((root / "wheels.json").read_text())["packages"]}
    versions.update(loggetta="34ecb6cec6f43a6f8607ff9f192749fdc7b587e9", **{"experts4bit-qlora": marker["source"]})
    tokens = json.loads((root / "locked_inputs.json").read_text())["tokens"]
    values = {
        text: {"ppl": 2.0, "nll": math.log(2) * 16376, "targets": 16376}
        for text in ("alpaca-heldout", "wikitext-test")
    }
    slots = {f"cpu-fixture-slot-{i}": hashlib.sha256(str(i).encode()).hexdigest() for i in range(448)}
    rows = {}
    for arm in ("L", "U", "U0"):
        row = {
            "schema": quality.SCHEMA,
            "science_eligible": False,
            "arm": arm,
            "source": marker["source"],
            "nonce": marker["nonce"],
            "runtime": versions,
            "tokens": tokens,
            "initial": slots,
            "optimizer_updates": 0,
            "frozen_unchanged": True,
            "path_before": {"fixture": arm},
            "path_after": {"fixture": arm},
            "base": {
                "cpu-fixture": {"dtype": "torch.bfloat16", "shape": [1], "values_sha256": slots["cpu-fixture-slot-0"]}
            },
            "provenance": {
                name: file_sha(root / name)
                for name in (
                    "science.sha256",
                    "wheels.json",
                    "model_files.json",
                    "locked_inputs.json",
                    "source_authority.json",
                )
            },
            "initial_quality": copy.deepcopy(values),
        }
        from dq11_proof_policy import SHIPPED_POLICY

        row["execution_settings"] = copy.deepcopy(SHIPPED_POLICY)
        row["engaged"] = {
            "a2000_initial_quality": {
                "schema": quality.SCHEMA,
                "science_eligible": False,
                "source": marker["source"],
                "nonce": marker["nonce"],
                "gpu": "NVIDIA RTX A2000 12GB",
                "cuda": "12.8",
                "sm": [8, 6],
                "optimizer_updates": 0,
                "total_bytes": 12 * (1 << 30),
                "free_before_bytes": 10 * (1 << 30),
                "process_cap_bytes": 9 * (1 << 30),
            }
        }
        row["identity_sha256"] = object_sha({"initial": slots, "tokens": tokens, "runtime": versions})
        rows[arm] = row
    return rows


def write_initial_rows(root, rows):
    for arm, row in rows.items():
        (root / f"receipts/initial-{arm}.json").write_text(json.dumps(row))


def test_original_k8_is_rederived_and_never_gives_science_or_paid_licence(sealed):
    import math
    from dq11_reduce import reduce

    root, marker = sealed
    rows = initial_rows_fixture(root, marker)
    write_initial_rows(root, rows)
    result = quality.report(root)
    assert result["status"] == "INITIAL_QUALITY_PASS" and result["U0_status"] == "complete"
    assert result["science_eligible"] is result["paid_GO"] is False
    assert result["recommendation"] is None and result["optimizer_updates"] == 0
    q = rows["U"]["initial_quality"]["alpaca-heldout"]
    q.update(ppl=2.2, nll=math.log(2.2) * 16376)
    write_initial_rows(root, rows)
    result = quality.report(root)
    assert result["status"] == "INITIAL_QUALITY_FAIL"
    assert result["comparisons"][0]["k8_pass"] is False
    assert result["comparisons"][0]["texts"][0]["absolute_ppl_delta"] > 0.05
    assert reduce(list(rows.values()), [], initial_only=True)["verdict"] == "VOID"
    (root / "receipts/initial-U0.json").unlink()
    assert quality.report(root)["U0_status"] == "not_included"


@pytest.mark.parametrize(
    "mutation",
    [
        "science",
        "updates",
        "source",
        "nonce",
        "runtime",
        "tokens",
        "initializer",
        "frozen",
        "binding",
        "provenance",
        "identity",
        "nan",
        "nll",
        "targets",
        "policy",
        "hardware",
    ],
)
def test_initial_only_report_refuses_corrupt_matched_evidence(sealed, mutation):
    root, marker = sealed
    rows = initial_rows_fixture(root, marker)
    row = rows["U"]
    if mutation == "science":
        row["science_eligible"] = True
    elif mutation == "updates":
        row["optimizer_updates"] = 1
    elif mutation == "source":
        row["source"] = "1" * 40
    elif mutation == "nonce":
        row["nonce"] = "other"
    elif mutation == "runtime":
        row["runtime"] = dict(row["runtime"], torch="2.12.1+cu130")
    elif mutation == "tokens":
        row["tokens"] = {}
    elif mutation == "initializer":
        row["initial"] = {}
    elif mutation == "frozen":
        row["base"] = {"changed": {"dtype": "torch.bfloat16", "shape": [1], "values_sha256": "1" * 64}}
    elif mutation == "binding":
        row["path_after"] = {}
    elif mutation == "provenance":
        row["provenance"] = {}
    elif mutation == "identity":
        row["identity_sha256"] = "1" * 64
    elif mutation == "policy":
        row["execution_settings"]["tf32_matmul"] = True
    elif mutation == "hardware":
        row["engaged"]["a2000_initial_quality"]["sm"] = [12, 0]
    elif mutation == "nan":
        row["initial_quality"]["alpaca-heldout"]["ppl"] = float("nan")
    elif mutation == "nll":
        row["initial_quality"]["alpaca-heldout"]["nll"] += 1
    else:
        row["initial_quality"]["alpaca-heldout"]["targets"] -= 1
    write_initial_rows(root, rows)
    with pytest.raises(ValueError):
        quality.report(root)


@pytest.mark.parametrize("failure", [None, "initial-U", "initial-U0", "optional-U0-report"])
def test_owned_pipeline_preserves_required_result_before_optional_work(sealed, monkeypatch, failure):
    """Exercise the actual controller, replacing only staging and subprocesses."""
    import subprocess

    import dq11_rehearsal
    import dq11_science_stage

    assets, marker = sealed
    monkeypatch.setenv("DQ11_GPU_CLAIM_CONFIRMED", "1")
    monkeypatch.setenv("TC1_DEADLINE_EPOCH", "2000000000")
    # External canonical assets are synthetic here; actual stage validation has
    # its own corruption coverage. Use real manifests and runtime lock bytes.
    staged = [
        LANE / name
        for name in ("model_files.json", "locked_inputs.json", "wheels.json", "requirements.lock", "science.sha256")
    ]
    for name in ("tokens.json", "adapter_init.safetensors"):
        (assets / name).write_text("CPU fixture, never used by a model")
        staged.append(assets / name)
    monkeypatch.setattr(dq11_science_stage, "stage", lambda *args: staged)
    output = assets / "owned-run"
    phases = []

    def external_phase(directory, env, name, cap, *command):
        phases.append(name)
        assert env["TC1_DEADLINE_EPOCH"] == "2000000000"
        ledger = json.loads((directory / "receipts/phase-timings.json").read_text())
        assert ledger["reserve_seconds"] == 300 and ledger["science_eligible"] is False
        if name.startswith("initial-U0") or name == "optional-U0-report":
            assert (directory / "receipts/initial-quality-primary.json").read_text() == "primary result\n"
        if name == failure:
            raise subprocess.TimeoutExpired(command, cap)
        if name == "report":
            assert "--primary-only" in command
            (directory / "receipts/initial-quality-primary.json").write_text("primary result\n")
        if name == "optional-U0-report":
            (directory / "receipts/initial-quality-with-U0.json").write_text("optional result\n")

    monkeypatch.setattr(dq11_rehearsal, "timed_phase", external_phase)
    if failure == "initial-U":
        with pytest.raises(subprocess.TimeoutExpired):
            quality.run(output, assets)
        assert "report" not in phases and "initial-U0" not in phases
        expected = "REFUSED\n"
    else:
        quality.run(output, assets)
        assert phases[:7] == [
            "git-prerequisite",
            "create-venv",
            "bootstrap",
            "prepare-real-model",
            "initial-L",
            "initial-U",
            "report",
        ]
        assert (output / "receipts/initial-quality-primary.json").read_text() == "primary result\n"
        expected = "COMPLETE_INITIAL_ONLY\n"
        if failure:
            refusal = json.loads((output / "receipts/optional-U0-refusal.json").read_text())
            assert refusal["science_eligible"] is False
    assert (output / ("QUALITY_STATUS." + marker["nonce"])).read_text() == expected
    assert (output / ("QUALITY_DONE." + marker["nonce"])).exists()
    assert not (output / "SUCCESS.json").exists()
    with pytest.raises(FileExistsError):
        quality.run(output, assets)
    assert (output / ("QUALITY_STATUS." + marker["nonce"])).read_text() == expected


@pytest.mark.parametrize("gradient", [False, True])
@pytest.mark.parametrize("selected_arm", ["L", "U", "U0"])
def test_worker_calls_original_preparation_initializer_and_scorer_without_updates(
    sealed, monkeypatch, gradient, selected_arm
):
    from contextlib import nullcontext

    import dq11_arm
    import dq11_common
    import dq11_proof_policy

    root, marker = sealed
    rows = initial_rows_fixture(root, marker)
    row = rows[selected_arm]
    monkeypatch.setenv("DQ11_GPU_CLAIM_CONFIRMED", "1")
    model = object()
    tokens = {"alpaca-heldout": [[1, 0, 2], [0, 3, 4]], "wikitext-test": [[1, 2, 3]]}
    calls = []

    def prepare(directory, arm, *, quality_read):
        assert directory == root and arm == selected_arm and quality_read is True
        calls.append("prepare")
        return model, None, None, row["engaged"]

    def install_initial(actual, path):
        assert actual is model and path == root / "adapters/adapter_init.safetensors"
        calls.append("install_initial")
        return row["initial"]

    def score(actual, actual_tokens):
        assert actual is model and actual_tokens is tokens
        calls.append("score")
        return row["initial_quality"]

    monkeypatch.setattr(dq11_arm, "prepare", prepare)
    monkeypatch.setattr(dq11_arm, "runtime", lambda directory: row["runtime"])
    monkeypatch.setattr(dq11_arm, "read_inputs", lambda directory: ({"tokens": row["tokens"]}, tokens))
    monkeypatch.setattr(dq11_common, "install_initial", install_initial)
    monkeypatch.setattr(dq11_common, "score", score)
    monkeypatch.setattr(dq11_common, "path_census", lambda *args: row["path_before"])
    monkeypatch.setattr(dq11_common, "base_census", lambda actual: row["base"])
    monkeypatch.setattr(
        dq11_common,
        "adapter_slots",
        lambda actual: ({"fixture": SimpleNamespace(grad=object() if gradient else None)}, {}),
    )
    monkeypatch.setattr(dq11_proof_policy, "process_environment", lambda kind: nullcontext())
    monkeypatch.setattr(dq11_proof_policy, "settings", lambda: row["execution_settings"])
    if gradient:
        with pytest.raises(ValueError, match="produced gradients"):
            quality.worker(root, selected_arm)
    else:
        result = quality.worker(root, selected_arm)
        assert result["optimizer_updates"] == 0 and result["science_eligible"] is False
        assert result["initial_quality"] == row["initial_quality"]
        if selected_arm in ("U", "U0"):
            assert result["engaged"]["token_0_scored_inputs"] == {"alpaca-heldout": 2, "wikitext-test": 0}
        else:
            assert "token_0_scored_inputs" not in result["engaged"]
    assert calls == ["prepare", "install_initial", "score"]
