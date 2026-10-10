"""Mandatory CPU instrument fixtures. These tests contain no scientific readings."""

import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlparse

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).parents[1]
LANE = ROOT / "bench/dq11"
sys.path.insert(0, str(LANE))
import dq11_common as common  # noqa: E402
import dq11_bootstrap as bootstrap  # noqa: E402
import dq11_launch as launch_controller  # noqa: E402
from dq11_observe import Observer  # noqa: E402
import dq11_reduce as reducer  # noqa: E402
from dq11_science_stage import stage  # noqa: E402


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fixture_receipts():
    """Synthetic values with actual registration metadata; never written as run evidence."""
    slots = {
        f"layers.{layer}.{group}.{projection}.{suffix}": sha(b"fixture initial")
        for layer in range(32)
        for group, projections in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for projection in projections
        for suffix in ("A", "B")
    }
    versions = {p["name"]: p["version"] for p in json.loads((LANE / "wheels.json").read_text())["packages"]}
    # This is a synthetic source value, never an asserted Git object.
    versions.update(loggetta="34ecb6cec6f43a6f8607ff9f192749fdc7b587e9", **{"experts4bit-qlora": "0" * 40})
    tokens = json.loads((LANE / "locked_inputs.json").read_text())["tokens"]
    quality = {name: {"ppl": 2.0, "targets": 16376, "nll": math.log(2) * 16376} for name in reducer.TEXTS}
    proofs = []
    for arm in ("L", "U", "U0"):
        routes = (
            [
                f"{prefix}:{layer}"
                for layer in range(32)
                for prefix in ("qkv", "o", "mlp", "LoRA_QKV", "LoRA_W", "LoRA_MLP")
            ]
            if arm == "U"
            else [key.removesuffix(".A") for key in slots if key.endswith(".A")]
        )
        dtype = "torch.bfloat16" if arm == "U" else "torch.float32"
        provenance = {
            name: common.file_sha(LANE / name) for name in ("science.sha256", "wheels.json", "model_files.json")
        }
        provenance["source_authority.json"] = sha(b"fixture authority")
        row = dict(
            kind="proof",
            arm=arm,
            repetition=0,
            nonce="fixture-no-instance",
            runtime=versions,
            tokens=tokens,
            input_seal_sha256=common.file_sha(LANE / "locked_inputs.json"),
            initial=slots,
            base={"fixture.weight": {"dtype": "torch.bfloat16", "shape": [2], "values_sha256": sha(b"fixture base")}},
            provenance=provenance,
            path_before={"fixture": arm},
            path_after={"fixture": arm},
            observer_same_arm_bitwise=True,
            frozen_unchanged=True,
            initial_quality=quality,
            proof_gradients_sha256=slots,
            execution={
                "calls": dict.fromkeys(routes, 2),
                "observer_removed": True,
                "gradients": {key: {"dtype": "torch.float32", "shape": [16, 2]} for key in slots},
                "operations": [
                    {
                        "route": route,
                        "operator": "aten.mm.default",
                        "result_dtype": dtype,
                        "operands": [{"dtype": dtype, "shape": [2, 16]}, {"dtype": dtype, "shape": [16, 2]}],
                    }
                    for route in routes
                    if not route.startswith("LoRA_")
                ],
            },
        )
        row["identity_sha256"] = common.object_sha({"initial": slots, "tokens": tokens, "runtime": versions})
        proofs.append(copy.deepcopy(row))
    reads = []
    for rep, arm in reducer.ORDER:
        row = copy.deepcopy(next(p for p in proofs if p["arm"] == arm))
        row.update(kind="read", repetition=rep, final_quality=copy.deepcopy(quality), final_adapters_sha256=slots)
        seconds = {"L": 2.0, "U": 1.0, "U0": 1.1}[arm]
        row["training"] = {
            "status": "OK",
            "correctness": {
                "all_finite": True,
                "frozen_dense_bytes_unchanged": True,
                "adapters_moved": True,
                "steps_without_trained_tokens": 0,
                "losses": [1.0] * 40,
            },
            "measured": {"step_seconds": [seconds] * 40, "timed_steps": "6..40"},
        }
        reads.append(row)
    teardown = {
        "complete": True,
        "instance_id": "fixture-no-instance",
        "evidence": {
            "instance_absent": True,
            "destroy": {"instance_id": "fixture-no-instance", "http": 200},
            "list_after": [],
        },
    }
    return proofs, reads, teardown


def test_registered_reducer_requires_teardown_before_recommendation():
    proofs, reads, teardown = fixture_receipts()
    assert reducer.reduce(proofs, [], initial_only=True)["verdict"] == "INITIAL_GATE_PASS"
    pending = reducer.reduce(proofs, reads)
    assert pending["verdict"] == "AWAITING_TEARDOWN" and pending["recommendation"] is None
    result = reducer.reduce(proofs, reads, teardown=teardown, instance_id="fixture-no-instance")
    assert result["verdict"] == "VALID_CONTROLLED_READ" and result["recommendation"] == "BUILD_CANDIDATE"
    assert result["scope"] == reducer.SCOPE and result["default_unsloth_position"] is False
    assert result["capacity_licensed"] is result["shipping_licensed"] is False
    assert len(result["pairs"]) == 2


@pytest.mark.parametrize("phase", ["initial", "trained"])
def test_finite_quality_failure_cannot_license_speed(phase):
    proofs, reads, teardown = fixture_receipts()
    rows, field = (proofs, "initial_quality") if phase == "initial" else (reads, "final_quality")
    row = next(row for row in rows if row["arm"] == "U0")
    q = row[field]["alpaca-heldout"]
    q.update(ppl=2.2, nll=math.log(2.2) * 16376)
    result = reducer.reduce(proofs, reads, teardown=teardown, instance_id="fixture-no-instance")
    assert result["verdict"] == "QUALITY_FAIL" and result["recommendation"] is None
    assert result["phase"] == phase


@pytest.mark.parametrize(
    "mutation",
    [
        "runtime",
        "tokens",
        "nonce",
        "observer",
        "binding",
        "gradient",
        "precision",
        "partial_calls",
        "closure",
        "identity",
        "base",
        "ppl",
        "nll",
        "targets",
        "missing_read",
        "order",
        "steps",
        "skip",
        "timing",
        "loss",
        "final_keys",
        "frozen",
        "teardown",
    ],
)
def test_reducer_voids_invalid_or_partial_evidence(mutation):
    proofs, reads, teardown = fixture_receipts()
    row = proofs[0]
    if mutation == "runtime":
        row["runtime"]["torch"] = "changed"
    elif mutation == "tokens":
        row["tokens"]["train"]["sha256"] = sha(b"changed")
    elif mutation == "nonce":
        row["nonce"] = "other-fixture"
    elif mutation == "observer":
        row["execution"]["observer_removed"] = False
    elif mutation == "binding":
        row["path_after"] = {"changed": True}
    elif mutation == "gradient":
        row["proof_gradients_sha256"].pop(next(iter(row["initial"])))
    elif mutation == "precision":
        proofs[1]["execution"]["operations"][0]["operands"][0]["dtype"] = "torch.float32"
    elif mutation == "partial_calls":
        proofs[1]["execution"]["calls"].pop("LoRA_QKV:31")
    elif mutation == "closure":
        row["provenance"]["science.sha256"] = sha(b"changed")
    elif mutation == "identity":
        row["identity_sha256"] = sha(b"changed")
    elif mutation == "base":
        row["base"]["fixture.weight"]["values_sha256"] = sha(b"changed")
    elif mutation == "ppl":
        row["initial_quality"]["alpaca-heldout"]["ppl"] = float("nan")
    elif mutation == "nll":
        row["initial_quality"]["alpaca-heldout"]["nll"] += 1
    elif mutation == "targets":
        row["initial_quality"]["alpaca-heldout"]["targets"] -= 1
    elif mutation == "missing_read":
        reads.pop()
    elif mutation == "order":
        reads.reverse()
    elif mutation == "steps":
        reads[0]["training"]["correctness"]["losses"].pop()
    elif mutation == "skip":
        reads[0]["training"]["correctness"]["steps_without_trained_tokens"] = 1
    elif mutation == "timing":
        reads[0]["training"]["measured"]["step_seconds"][6] = 0
    elif mutation == "loss":
        reads[0]["training"]["correctness"]["losses"][5] = float("inf")
    elif mutation == "final_keys":
        reads[0]["final_adapters_sha256"].pop(next(iter(row["initial"])))
    elif mutation == "frozen":
        reads[0]["frozen_unchanged"] = False
    elif mutation == "teardown":
        teardown["evidence"]["list_after"] = ["fixture-no-instance"]
    result = reducer.reduce(proofs, reads, teardown=teardown, instance_id="fixture-no-instance")
    assert result["verdict"] == "VOID", result
    assert result["recommendation"] is None


def test_cutoff_requires_both_repetitions_and_warmups_are_excluded():
    proofs, reads, teardown = fixture_receipts()
    for row in reads:
        row["training"]["measured"]["step_seconds"][:5] = [999.0] * 5
        if row["repetition"] == 2 and row["arm"] == "U0":
            row["training"]["measured"]["step_seconds"][5:] = [1.01] * 35
    result = reducer.reduce(proofs, reads, teardown=teardown, instance_id="fixture-no-instance")
    assert result["recommendation"] == "DEFER_FUSION"
    assert result["pairs"][0]["median_seconds"]["U"] == 1


def test_missing_receipt_cli_emits_void(tmp_path):
    result = subprocess.run(
        [sys.executable, str(LANE / "dq11_reduce.py"), str(tmp_path)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 12 and json.loads(result.stdout)["verdict"] == "VOID"


@pytest.mark.parametrize("kind,rep", [("proof", 1), ("proof", 2), ("read", 0)])
def test_invalid_worker_phase_refuses_before_framework_loading(kind, rep):
    from types import SimpleNamespace
    from dq11_arm import run

    with pytest.raises(ValueError, match="phase/repetition"):
        run(SimpleNamespace(kind=kind, repetition=rep))


def fixture_stage(tmp_path):
    here, assets = tmp_path / "bench/dq11", tmp_path / "payload"
    here.mkdir(parents=True)
    assets.mkdir()
    (here / "subject.py").write_bytes(b"fixture source")
    (assets / "adapter_init.safetensors").write_bytes(b"fixture not a scientific adapter")
    (assets / "tokens.json").write_bytes(b"fixture not scientific tokens")
    locked = {"assets": {p.name: {"sha256": common.file_sha(p), "bytes": p.stat().st_size} for p in assets.iterdir()}}
    (here / "locked_inputs.json").write_text(json.dumps(locked))
    names = ("subject.py", "locked_inputs.json")
    (here / "science.sha256").write_text("".join(common.file_sha(here / name) + "  " + name + "\n" for name in names))
    return here, assets


def test_external_assets_checked_before_flat_transport(tmp_path):
    here, assets = fixture_stage(tmp_path)
    paths = stage(here, assets)
    assert len(paths) == 5 and len({p.name for p in paths}) == 5
    (assets / "tokens.json").write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="canonical asset"):
        stage(here, assets)


@pytest.mark.parametrize("mutation", ["missing", "changed", "duplicate", "path", "empty", "ambiguous_asset"])
def test_stage_refuses_corrupt_closure_and_assets(tmp_path, mutation):
    here, assets = fixture_stage(tmp_path)
    manifest = here / "science.sha256"
    if mutation == "missing":
        (here / "subject.py").unlink()
    elif mutation == "changed":
        (here / "subject.py").write_bytes(b"changed")
    elif mutation == "duplicate":
        manifest.write_text(manifest.read_text() * 2)
    elif mutation == "path":
        manifest.write_text(manifest.read_text().replace("subject.py", "../subject.py"))
    elif mutation == "empty":
        manifest.write_text("")
    elif mutation == "ambiguous_asset":
        (assets / "data").mkdir()
        shutil.copy(assets / "tokens.json", assets / "data/tokens.json")
    with pytest.raises(ValueError):
        stage(here, assets)


def test_runtime_code_mutation_cannot_pass_source_identity(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("def function(x):\n    return x + 1\n")
    spec = importlib.util.spec_from_file_location("dq11_fixture_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    authority = {common.file_sha(path)}
    common.verify_callable(module.function, authority)
    module.function.__code__ = module.function.__code__.replace(co_consts=(None, 2))
    with pytest.raises(ValueError, match="runtime code"):
        common.verify_callable(module.function, authority)


@pytest.mark.parametrize("mutation", ["callee_code", "callee_alias"])
def test_unchanged_caller_cannot_hide_changed_global_callee(tmp_path, mutation):
    path = tmp_path / "global_source.py"
    path.write_text(
        "def helper(x):\n    return x + 1\ndef other(x):\n    return x + 2\ndef caller(x):\n    return helper(x)\n"
    )
    spec = importlib.util.spec_from_file_location("dq11_global_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    authority = {common.file_sha(path)}
    expected = {"helper": (module, "helper")}
    common.verify_named_callees(module, expected, authority)
    caller_identity = common.verify_callable(module.caller, authority)
    assert module.caller(3) == 4
    if mutation == "callee_code":
        replacement = compile("def helper(x):\n    return x + 2\n", str(path), "exec")
        module.helper.__code__ = next(c for c in replacement.co_consts if isinstance(c, type(module.helper.__code__)))
    else:
        module.helper = module.other
    # The old caller-only check still passes, while executed semantics changed.
    assert common.verify_callable(module.caller, authority) == caller_identity
    assert module.caller(3) == 5
    with pytest.raises(ValueError, match="runtime code|callee binding"):
        common.verify_named_callees(module, expected, authority)


@pytest.mark.parametrize("mutation", ["src", "fn"])
def test_changed_jit_source_or_named_function_refuses(tmp_path, mutation):
    from types import SimpleNamespace

    path = tmp_path / "kernel_source.py"
    path.write_text("def kernel(x):\n    return x + 1\ndef other(x):\n    return x + 2\n")
    spec = importlib.util.spec_from_file_location("dq11_kernel_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = SimpleNamespace(fn=module.kernel, src="def kernel(x):\n    return x + 1\n")
    authority = {common.file_sha(path)}
    common.verify_jit_source(kernel, authority, defining=module, qualname="kernel")
    if mutation == "src":
        kernel.src = kernel.src.replace("x + 1", "x + 2")
    else:
        kernel.fn = module.other
    with pytest.raises(ValueError, match="JIT source|callee binding"):
        common.verify_jit_source(kernel, authority, defining=module, qualname="kernel")


@pytest.mark.parametrize("decorator", ["@torch.inference_mode", "@torch.inference_mode()"])
def test_inference_decorator_factory_mode_is_sealed(tmp_path, decorator):
    import inspect
    import torch.autograd.grad_mode as grad_mode
    import torch.utils._contextlib as contextlib

    path = tmp_path / "context_source.py"
    path.write_text("import torch\n" + decorator + "\ndef function(x):\n    return x + 1\n")
    spec = importlib.util.spec_from_file_location("dq11_context_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    authority = {
        common.file_sha(p) for p in (path, inspect.getsourcefile(grad_mode), inspect.getsourcefile(contextlib))
    }
    common.verify_callable(module.function, authority)
    factory = next(cell.cell_contents for cell in module.function.__closure__ if inspect.ismethod(cell.cell_contents))
    factory.__self__.mode = False
    with pytest.raises(ValueError, match="context factory"):
        common.verify_callable(module.function, authority)


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 16, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(16, 2, bias=False)})
        self.base = nn.Linear(2, 2, bias=False)
        self.base.weight.requires_grad_(False)
        self.r, self.lora_alpha, self.scaling, self.use_dora = (
            {"default": 16},
            {"default": 32},
            {"default": 2.0},
            {"default": False},
        )

    def get_base_layer(self):
        return self.base

    def forward(self, x):
        return self.base(x) + self.lora_B["default"](self.lora_A["default"](x)) * 2


def small_model():
    model = nn.Module()
    model.layers = nn.ModuleList()
    for _ in range(32):
        layer = nn.Module()
        layer.self_attn, layer.mlp = nn.Module(), nn.Module()
        for owner, names in (
            (layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
            (layer.mlp, ("gate_proj", "up_proj", "down_proj")),
        ):
            for name in names:
                setattr(owner, name, Projection())
        model.layers.append(layer)
    return model


def test_actual_dispatch_observer_with_nonreentrant_checkpoint_removes_all_hooks():
    torch.manual_seed(4)
    model = small_model()
    slots, modules = common.adapter_slots(model)
    x = torch.ones(2, 2, requires_grad=True)

    def forward(x):
        return sum(module(x).sum() for module in modules.values())

    observer = Observer(model, "L")
    with observer.active():
        checkpoint(forward, x, use_reentrant=False).backward()
    observed = {key: p.grad.clone() for key, p in slots.items()}
    receipt = observer.receipt()
    assert receipt["observer_removed"] and len(receipt["gradients"]) == 448
    assert all(receipt["calls"][key] >= 1 for key in modules)
    assert all(not p._backward_hooks for p in slots.values())
    assert all("forward" not in module.__dict__ for module in modules.values())
    model.zero_grad(set_to_none=True)
    checkpoint(forward, x, use_reentrant=False).backward()
    for key, p in slots.items():
        assert torch.equal(observed[key], p.grad)


def test_fused_backward_route_does_not_unpack_checkpoint_saved_tensors_twice():
    weight = torch.ones(2, 2)

    class FusedFixture(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x, weight)
            ctx.custom_saved_tensors = (weight, None, 2)
            return x @ weight

        @staticmethod
        def backward(ctx, gradient):
            _, w = ctx.saved_tensors
            return gradient @ w.T

    observer = Observer(nn.Module(), "U")
    observer.parameter_layers = {weight.data_ptr(): 7}
    descriptor = FusedFixture.__dict__["backward"]
    observer.wrap(FusedFixture, "backward", "LoRA_W", backward=True)
    try:
        x = torch.ones(2, 2, requires_grad=True)
        checkpoint(FusedFixture.apply, x, use_reentrant=False).sum().backward()
        assert observer.calls["LoRA_W:7"] == 1 and torch.equal(x.grad, torch.full_like(x, 2))
    finally:
        owner, name, _, original = observer.restore.pop()
        setattr(owner, name, original)
    assert FusedFixture.__dict__["backward"] is descriptor


def test_observer_cleanup_on_exception():
    model = small_model()
    slots, modules = common.adapter_slots(model)
    observer = Observer(model, "L")
    with pytest.raises(RuntimeError, match="fixture failure"):
        with observer.active():
            raise RuntimeError("fixture failure")
    assert observer.removed and all(not p._backward_hooks for p in slots.values())
    assert all("forward" not in module.__dict__ for module in modules.values())


def box_fixture(tmp_path, monkeypatch, failure, *, elapsed=None, guard_seconds=7200):
    """Run the actual shell with clearly synthetic external commands, no GPU/network."""
    work, commands = tmp_path / "work", tmp_path / "bin"
    work.mkdir()
    commands.mkdir()
    shutil.copy(LANE / "dq11_require_git.sh", work)
    (work / "fixture-clock").write_text("0")
    for name in ("adapter_init.safetensors", "tokens.json"):
        (work / name).write_bytes(b"fixture payload")
    (work / "science.sha256").write_text(sha(b"fixture payload") + "  tokens.json\n")
    (work / "inputs.sha256").write_text(sha(b"fixture payload") + "  data/tokens.json\n")
    program = (
        f"#!{sys.executable}\n"
        + """import hashlib,json,os,sys
from pathlib import Path
name=Path(sys.argv[0]).name; args=sys.argv[1:]; fail=os.environ['FIXTURE_FAIL']
with Path('calls.txt').open('a') as f:f.write(name+' '+' '.join(args)+'\\n')
if name=='sha256sum':
 for line in Path(args[1]).read_text().splitlines():
  h,p=line.split(); assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
elif name=='nvidia-smi':print('NVIDIA GeForce RTX 5090, 32607')
elif name=='df':print('Filesystem 1024-blocks Used Available Capacity Mounted on\\nfixture 999999999 0 999999999 0% /')
elif name=='date':
 if args==['+%s']:print(Path('fixture-clock').read_text())
 else:os.execv(os.environ['FIXTURE_REAL_DATE'],[os.environ['FIXTURE_REAL_DATE'],*args])
elif name=='perl':
 now=int(Path('fixture-clock').read_text()); left=int(os.environ['TC1_DEADLINE_EPOCH'])-now-300
 with Path('alarms.jsonl').open('a') as f:f.write(json.dumps({'command':args[3:],'cap':int(args[2]),'left':left})+'\\n')
 advance=json.loads(os.environ.get('FIXTURE_ELAPSED','{}')).get(args[4] if len(args)>4 else args[3],0)
 Path('fixture-clock').write_text(str(now+advance))
 os.execvp(args[3],args[3:])
elif name=='timeout':os.execvp(args[1],args[1:])
elif name=='apt-get':sys.exit(5)
elif args and args[0]=='dq3_vram_probe.py':sys.exit(3 if fail=='vram' else 0)
elif args and args[0]=='dq3_egress_probe.py':sys.exit(4 if fail=='egress' else 0)
elif args[:2]==['-m','venv']:Path(args[2]+'/bin').mkdir(parents=True)
elif args and args[0]=='dq11_arm.py':
 kind=args[args.index('--kind')+1]; arm=args[args.index('--arm')+1]
 if fail=='proof' and kind=='proof' and arm=='U':sys.exit(11)
 Path(args[args.index('--out')+1]).write_text(json.dumps({'fixture':True,'arm':arm,'kind':kind}))
elif args and args[0]=='dq11_reduce.py':
 if fail=='initial' and '--initial-only' in args:sys.exit(12)
 print(json.dumps({'fixture':True,'scientific_evidence':False}))
"""
    )
    for name in (
        "sha256sum",
        "nvidia-smi",
        "df",
        "python3",
        "python3.11",
        "python",
        "perl",
        "timeout",
        "apt-get",
        "date",
    ):
        p = commands / name
        p.write_text(program)
        p.chmod(0o755)
    if failure == "git":
        for name in ("bash", "mkdir", "mv", "tee", "tail", "awk", "cat"):
            (commands / name).symlink_to(shutil.which(name))
        monkeypatch.setenv("PATH", str(commands))
    else:
        monkeypatch.setenv("PATH", str(commands) + os.pathsep + os.environ["PATH"])
    env = dict(
        os.environ,
        FIXTURE_FAIL=failure,
        DQ11_W=str(work),
        TC1_RUN_NONCE="fixture-science",
        TC1_DEADLINE_EPOCH=str(1 if failure == "deadline" else guard_seconds),
        FIXTURE_REAL_DATE=shutil.which("date", path=os.defpath),
        FIXTURE_ELAPSED=json.dumps(elapsed or {}),
        E4B_SHA="0" * 40,
    )
    result = subprocess.run(
        ["bash", str(LANE / "dq11_science_run.sh")], env=env, capture_output=True, text=True, timeout=30
    )
    return result, work, (work / "calls.txt").read_text()


@pytest.mark.parametrize(
    "failure,rc",
    [("git", 20), ("vram", 18), ("egress", 14), ("proof", 11), ("initial", 12), ("deadline", 11), ("none", 0)],
)
def test_actual_box_shell_phase_order_refusals_and_nonce_markers(tmp_path, monkeypatch, failure, rc):
    result, work, calls = box_fixture(tmp_path, monkeypatch, failure)
    assert result.returncode == rc, result.stdout + result.stderr
    assert (work / "TC1_EXIT_CODE.fixture-science").read_text().strip() == str(rc)
    assert (work / "TP_DONE.fixture-science").exists()
    assert (work / "TC1_SUCCESS.fixture-science").exists() == (rc == 0)
    if rc:
        assert "--kind read" not in calls
        if failure == "git":
            assert "apt-get update" in calls
            assert "dq3_vram_probe" not in calls and "dq3_egress_probe" not in calls
            assert "dq11_bootstrap" not in calls and "dq11_prepare" not in calls
    else:
        assert sum(line.startswith("python dq11_arm.py --kind read") for line in calls.splitlines()) == 6
        assert calls.index("--initial-only") < calls.index("--kind read")
        assert "PROVISIONAL" in result.stdout


def test_every_staged_deadline_cap_fits_two_hour_guard_after_install():
    run = (LANE / "dq11_science_run.sh").read_text()
    assert (
        dict(line.split() for line in (LANE / "science.sha256").read_text().splitlines())[
            sha((LANE / "dq11_science_run.sh").read_bytes())
        ]
        == "dq11_science_run.sh"
    )
    assert "for two hours" in (LANE / "DQ11-AMENDMENT-3.md").read_text()
    reserve = int(re.search(r"TC1_DEADLINE_EPOCH - \$\(date \+%s\) - (\d+)", run)[1])
    assert reserve == 300
    caps = [int(n) for n in re.findall(r"^budget_cap (\d+)$", run, re.M)]
    caps += [int(n) for n in re.findall(r'^\s*phase (?:(?:"[^"\n]+")|(?:[\w-]+)) (\d+) ', run, re.M)]
    assert sorted(caps) == [60, 120, 120, 240, 900, 900, 1800, 1800]
    # P109 pattern: each cap + fetch/teardown margin must fit even after
    # a conservative 900 s install/startup allowance, versus 210 s observed CPU.
    assert all(cap + reserve <= 2 * 3600 - 900 for cap in caps)
    alarms = re.findall(r"^perl -e 'alarm shift; exec @ARGV' (\S+)", run, re.M)
    alarms += re.findall(r"^  perl -e 'alarm shift; exec @ARGV' (\S+)", run, re.M)
    assert alarms == ['"$PHASE_CAP"'] * 4
    assert 'budget_cap "$requested"' in run
    assert '[ "$left" -ge "$requested" ] || PHASE_CAP=$left' in run


@pytest.mark.parametrize("bootstrap_elapsed,expected_rc", [(210, 0), (1800, 0), (6400, 0), (6500, 11)])
def test_actual_runner_clamps_all_alarms_after_install_and_preserves_reserve(
    tmp_path, monkeypatch, bootstrap_elapsed, expected_rc
):
    result, work, calls = box_fixture(
        tmp_path,
        monkeypatch,
        "none",
        elapsed={
            "dq11_require_git.sh": 240,
            "dq3_vram_probe.py": 120,
            "dq3_egress_probe.py": 60,
            "dq11_bootstrap.py": bootstrap_elapsed,
        },
    )
    assert result.returncode == expected_rc, result.stdout + result.stderr
    alarms = [json.loads(line) for line in (work / "alarms.jsonl").read_text().splitlines()]
    assert all(0 < row["cap"] <= row["left"] for row in alarms)
    assert alarms[0]["cap"] == 240
    if bootstrap_elapsed == 6400:
        assert next(row for row in alarms if "dq11_prepare.py" in row["command"])["cap"] == 80
    if bootstrap_elapsed == 6500:
        assert "dq11_prepare.py" not in calls and "--kind read" not in calls
    assert (work / "TC1_SUCCESS.fixture-science").exists() == (expected_rc == 0)


@pytest.mark.parametrize("guard_seconds,caps", [(310, [10, 10, 10]), (360, [60, 60, 60])])
def test_actual_runner_clamps_git_and_both_probes_near_deadline(tmp_path, monkeypatch, guard_seconds, caps):
    result, work, _ = box_fixture(tmp_path, monkeypatch, "none", guard_seconds=guard_seconds)
    assert result.returncode == 0, result.stdout + result.stderr
    alarms = [json.loads(line) for line in (work / "alarms.jsonl").read_text().splitlines()]
    assert [row["cap"] for row in alarms[:3]] == caps
    assert all(0 < row["cap"] <= row["left"] for row in alarms)


@pytest.mark.parametrize("installer", ["fails", "false_success", "supplies_git"])
def test_git_tool_step_is_bounded_and_checks_actual_availability(tmp_path, installer):
    commands = tmp_path / "bin"
    commands.mkdir()
    (commands / "timeout").write_text('#!/bin/bash\necho "$*" >> "$FIXTURE_CALLS"\nshift\nexec "$@"\n')
    (commands / "env").symlink_to(shutil.which("env"))
    (commands / "apt-get").write_text(
        f"#!{sys.executable}\n"
        "import os,sys\nfrom pathlib import Path\n"
        "if sys.argv[1]=='update':sys.exit(0)\n"
        "if os.environ['FIXTURE_INSTALLER']=='fails':sys.exit(7)\n"
        "if os.environ['FIXTURE_INSTALLER']=='supplies_git':\n"
        " p=Path(os.environ['PATH'])/'git';p.write_text('#!/bin/sh\\nexit 0\\n');p.chmod(0o755)\n"
    )
    for name in ("timeout", "apt-get"):
        (commands / name).chmod(0o755)
    calls = tmp_path / "calls"
    result = subprocess.run(
        [shutil.which("bash"), str(LANE / "dq11_require_git.sh")],
        env=dict(os.environ, PATH=str(commands), FIXTURE_INSTALLER=installer, FIXTURE_CALLS=str(calls)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (0 if installer == "supplies_git" else 20)
    assert calls.read_text().splitlines() == [
        "120 apt-get update",
        "120 env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git",
    ]


def test_actual_controller_refuses_dirty_unreviewed_source_before_transport(tmp_path):
    # Isolated Git fixture avoids depending on the test runner checkout's dirty state.
    repo = tmp_path / "repo"
    lane = repo / "bench/dq11"
    lane.mkdir(parents=True)
    shutil.copy(LANE / "dq11_science_drive.sh", lane)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    (repo / "dirty").write_text("fixture dirty")
    env = dict(os.environ, TC1_DRIVE_DRYRUN="0")
    result = subprocess.run(
        ["bash", str(lane / "dq11_science_drive.sh")], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 78 and "dirty source" in result.stdout
    (repo / "dirty").unlink()
    result = subprocess.run(
        ["bash", str(lane / "dq11_science_drive.sh")], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 78 and "unmerged instrument" in result.stdout


def test_committed_science_closure_and_selected_wheel_lock():
    names = []
    for line in (LANE / "science.sha256").read_text().splitlines():
        digest, name = line.split()
        names.append(name)
        matches = [root / name for root in (LANE, LANE.parent / "dq1", LANE.parent / "dq3") if (root / name).is_file()]
        assert len(matches) == 1 and common.file_sha(matches[0]) == digest
    assert len(names) == len(set(names)) and "science.sha256" not in names
    packages = json.loads((LANE / "wheels.json").read_text())["packages"]
    lock = (LANE / "requirements.lock").read_text()
    assert len(packages) == 101
    for package in packages:
        assert package["url"] in lock and "--hash=sha256:" + package["sha256"] in lock


def test_every_committed_wheel_url_has_an_allowlisted_origin():
    packages = json.loads((LANE / "wheels.json").read_text())["packages"]
    lock_urls = [
        line.split(" @ ", 1)[1].split()[0]
        for line in (LANE / "requirements.lock").read_text().splitlines()
        if " @ " in line
    ]
    assert len(lock_urls) == len(packages) == 101
    assert set(lock_urls) == {row["url"] for row in packages}
    assert all(urlparse(url).scheme == "https" for url in lock_urls)
    bootstrap.validate_wheel_origins([{"url": url} for url in lock_urls])


@pytest.mark.parametrize(
    "missing", ["files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org", "pypi.nvidia.com"]
)
def test_missing_locked_origin_refuses_before_network_or_install(tmp_path, monkeypatch, missing):
    packages = json.loads((LANE / "wheels.json").read_text())["packages"]
    (tmp_path / "wheels.json").write_text(json.dumps({"packages": packages}))
    monkeypatch.setattr(bootstrap, "ALLOWED_WHEEL_ORIGINS", bootstrap.ALLOWED_WHEEL_ORIGINS - {missing})

    def forbidden(*args, **kwargs):
        raise AssertionError("origin refusal must precede network and installation")

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(bootstrap.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="unregistered wheel origin: " + missing):
        bootstrap.install(tmp_path, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


class WheelResponse(io.BytesIO):
    def __init__(self, status=206):
        super().__init__(b"wheel fixture")
        self.status = status


@pytest.mark.parametrize("probe", [False, True])
def test_shared_fetch_uses_honest_ua_for_probe_and_download(monkeypatch, probe):
    def open_fixture(request, timeout):
        assert request.get_header("User-agent") == "dq11-bootstrap/1"
        assert request.get_header("Range") == ("bytes=0-0" if probe else None)
        assert timeout == 60
        return WheelResponse()

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", open_fixture)
    with bootstrap.fetch_wheel("https://download-r2.pytorch.org/fixture.whl", probe=probe) as response:
        assert bootstrap.read_wheel(response, "https://download-r2.pytorch.org/fixture.whl", 1) == b"w"


@pytest.mark.parametrize("status", [403, 503, 302])
def test_fetch_refusal_names_exact_url_and_status(monkeypatch, status):
    url = "https://download-r2.pytorch.org/fixture.whl"

    def refusal(request, timeout):
        if status != 302:
            raise HTTPError(request.full_url, status, "fixture refusal", {}, None)
        return WheelResponse(status)

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", refusal)
    with pytest.raises(bootstrap.WheelFetchError, match=f"URL={url} status={status}"):
        bootstrap.fetch_wheel(url)


def test_installer_uses_shared_fetch(monkeypatch, tmp_path):
    row = json.loads((LANE / "wheels.json").read_text())["packages"][0]
    (tmp_path / "wheels.json").write_text(json.dumps({"packages": [row]}))

    def refusal(url, *, probe=False):
        assert url == row["url"] and not probe
        raise bootstrap.WheelFetchError(url, 403, "shared opener fixture")

    monkeypatch.setattr(bootstrap, "fetch_wheel", refusal)
    with pytest.raises(bootstrap.WheelFetchError, match="shared opener fixture"):
        bootstrap.install(tmp_path, tmp_path / "cache")


def launch_fixture(tmp_path, monkeypatch):
    repo = tmp_path / "source"
    here, assets = fixture_stage(repo)
    for name in ("dq11_bootstrap.py", "dq11_launch.py", "wheels.json", "requirements.lock"):
        shutil.copy(LANE / name, here)
    names = (
        "subject.py",
        "locked_inputs.json",
        "dq11_bootstrap.py",
        "dq11_launch.py",
        "wheels.json",
        "requirements.lock",
    )
    (here / "science.sha256").write_text("".join(common.file_sha(here / name) + "  " + name + "\n" for name in names))
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(repo), "update-ref", "refs/remotes/origin/main", head], check=True)
    guard = tmp_path / "guard/tools"
    guard.mkdir(parents=True)
    marker = tmp_path / "guard-invoked"
    (guard / "pod-launch.sh").write_text('#!/bin/bash\nprintf "%s\\n" "$E4B_REPO" "$1" "$2" > "$FIXTURE_MARKER"\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"heads": {"e4b": head}}))
    monkeypatch.setenv("ADERTHA_REPO", str(guard.parent))
    monkeypatch.setenv("DQ11_ASSET_DIR", str(assets))
    monkeypatch.setenv("FIXTURE_MARKER", str(marker))
    return repo, manifest, tmp_path / "gate.json", marker


@pytest.mark.parametrize("failure,gate_only", [(None, False), ("torch", False), (None, True), ("torch", True)])
def test_actual_launch_controller_checks_all_urls_before_guard(tmp_path, monkeypatch, failure, gate_only):
    repo, manifest, output, marker = launch_fixture(tmp_path, monkeypatch)
    packages = json.loads((LANE / "wheels.json").read_text())["packages"]
    failing = next(row["url"] for row in packages if row["name"] == "torch")
    called = []

    def probe(request, timeout):
        called.append(request.full_url)
        assert request.get_header("User-agent") == "dq11-bootstrap/1"
        assert request.get_header("Range") == "bytes=0-0"
        if failure and request.full_url == failing:
            raise HTTPError(request.full_url, 403, "fixture refusal", {}, None)
        return WheelResponse()

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", probe)
    if failure:
        with pytest.raises(ValueError, match="BEFORE quote/rental"):
            launch_controller.launch(manifest, "fixture approval", output, repository=repo, gate_only=gate_only)
        assert not marker.exists()
    else:
        assert (
            launch_controller.launch(manifest, "fixture approval", output, repository=repo, gate_only=gate_only) == 0
        )
        if gate_only:
            assert not marker.exists()
        else:
            assert marker.read_text().splitlines() == [str(repo), str(manifest), "fixture approval"]
    assert called == [row["url"] for row in packages]
    report = json.loads(output.read_text())
    assert report["count"] == 101 and report["passed"] == (failure is None)
    assert len(report["results"]) == 101
    if failure:
        refused = [row for row in report["results"] if "error" in row]
        assert len(refused) == 1 and refused[0]["url"] == failing and refused[0]["status"] == 403


@pytest.mark.parametrize("mutation", ["dirty", "unmerged", "wrong_manifest", "changed_asset", "output_inside_source"])
def test_launch_source_refusals_precede_network_and_guard(tmp_path, monkeypatch, mutation):
    repo, manifest, output, marker = launch_fixture(tmp_path, monkeypatch)
    if mutation == "dirty":
        (repo / "dirty").write_text("fixture")
    elif mutation == "unmerged":
        subprocess.run(["git", "-C", str(repo), "update-ref", "-d", "refs/remotes/origin/main"], check=True)
    elif mutation == "wrong_manifest":
        manifest.write_text(json.dumps({"heads": {"e4b": "0" * 40}}))
    elif mutation == "changed_asset":
        # Keep source clean while replacing the external canonical fixture asset.
        assets = tmp_path / "external-assets"
        shutil.copytree(repo / "payload", assets)
        (assets / "tokens.json").write_text("changed")
        monkeypatch.setenv("DQ11_ASSET_DIR", str(assets))
    else:
        output = repo / "gate.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("source refusal must precede network")

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", forbidden)
    with pytest.raises(ValueError):
        launch_controller.launch(manifest, "fixture approval", output, repository=repo)
    assert not marker.exists() and not output.exists()


def test_actual_science_controller_dry_run_with_small_sealed_fixture(tmp_path):
    repo = tmp_path / "repo"
    here, assets = fixture_stage(repo)
    for name in ("dq11_science_drive.sh", "dq11_science_stage.py", "dq11_common.py", "dq11_science_run.sh"):
        shutil.copy(LANE / name, here)
    relatives = (
        "bench/dq1/adapter_path_audit.py",
        "bench/tc1/tc1_drive.sh",
        "bench/tc1/tc1_run.sh",
        "bench/tc1/tc1_arm.py",
        "bench/tc1/tc1_reduce.py",
        "bench/common/lane_liveness.sh",
        "bench/tp4/tp4_alpaca.py",
        "bench/flagship-matrix/drivers/n9_datasets.py",
        "bench/flagship-matrix/ds_manifest.json",
    )
    for relative in relatives:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, target)
    names = ("dq11_science_run.sh", "locked_inputs.json")
    (here / "science.sha256").write_text("".join(common.file_sha(here / name) + "  " + name + "\n" for name in names))
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    env = dict(
        os.environ,
        TC1_DRIVE_DRYRUN="1",
        DQ11_ASSET_DIR=str(assets),
        E4B_RENT_SSH_HOST="fixture.invalid",
        E4B_RENT_SSH_PORT="22",
        E4B_RENT_SSH_OPTS="-o BatchMode=yes",
        E4B_RENT_RUN_DIR=str(tmp_path),
        E4B_RENT_RUN_ID="fixture-no-run",
        E4B_RENT_DEADLINE_EPOCH=str(int(time.time()) + 7200),
        E4B_RENT_INSTANCE_ID="fixture-no-instance",
    )
    env.pop("E4B_SHA", None)
    result = subprocess.run(
        ["bash", str(here / "dq11_science_drive.sh")], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DRYRUN stage" in result.stdout and "bash dq11_science_run.sh" in result.stdout
    assert "adapter_init.safetensors tokens.json science.sha256" in result.stdout
