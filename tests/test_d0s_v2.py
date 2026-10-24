# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""bench/d0s/d0s_v2.py: a backend-adapter/2 manifest derived for a backend-adapter/1 fixture. CPU only.

A tiny trained fixture is built the way P224's fixture step builds one: the trained stacks are real Experts4bit stacks
inside ExpertsLoRA, stored as fixture-base stores them (codes and fp32 absmax, plus the double-quantized payload that
e4b's own compress_expert_absmax_ wrote), with a training receipt, a fixture_base.json record and a v1 adapter. The
digest is the trainer's computation (Loggetta's _expert_digest, reproduced here so CI needs no Loggetta).
"""
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

IN_CI = os.environ.get("CI") == "true"


def _need(name):
    return importlib.import_module(name) if IN_CI else pytest.importorskip(name)


torch = _need("torch")
_need("bitsandbytes")
_need("safetensors")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("d0s_v2", ROOT / "bench" / "d0s" / "d0s_v2.py")
V2 = importlib.util.module_from_spec(spec)
sys.modules["d0s_v2"] = V2
spec.loader.exec_module(V2)

L, E, INTER, H, R, ALPHA = 2, 3, 64, 128, 2, 4
REV = "c" * 40
STACK = "model.layers.{}.mlp.experts"
SETUP = {"r": R, "alpha": ALPHA, "quant_type": "nf4", "blocksize": 64, "attn_4bit": False, "train_attention": True,
         "train_experts": True}


def expert_digest(model, layers=(0, -1)):
    """Loggetta's _expert_digest (backends/experts4bit_train.py), statement for statement: the stored bytes of the
    packed codes and the absmax as held (fp32, or the double-quantized payload) of the named stacks."""
    from experts4bit_qlora import ExpertsNbit

    bases = [m for m in model.modules() if isinstance(m, ExpertsNbit)]
    out = {}
    for li in layers:
        base = bases[li]
        tensors = {n: getattr(base, n) for n in ("gate_up_proj", "down_proj")}
        for which in ("gate_up", "down"):
            if which + "_absmax_q" in base._buffers:
                tensors.update({which + "_absmax" + s: base._buffers[which + "_absmax" + s] for s in V2.SUFFIXES})
            else:
                tensors[which + "_absmax"] = getattr(base, which + "_absmax")
        sha = hashlib.sha256()
        for n in sorted(tensors):
            sha.update(tensors[n].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        out[f"stack[{li}]"] = sha.hexdigest()
    return out


def _repo(path, lora_text, tag="v0.56.1"):
    """A git repository holding experts4bit_qlora/lora.py at a release tag, the way the e4b repository holds it."""
    (path / "experts4bit_qlora").mkdir(parents=True)
    (path / "experts4bit_qlora" / "lora.py").write_text(lora_text)
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "lora"], ["tag", tag]):
        subprocess.run(["git", "-C", str(path), *args], check=True, env=env, capture_output=True)
    return path


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    W = {(li, e, p): (torch.randn(*s, generator=g) * 0.02).to(torch.bfloat16) for li in range(L) for e in range(E)
         for p, s in (("gate_proj", (INTER, H)), ("up_proj", (INTER, H)), ("down_proj", (H, INTER)))}
    A = {(li, p): (torch.randn(H, H, generator=g) * 0.02).to(torch.bfloat16) for li in range(L)
         for p in ("q_proj", "k_proj", "v_proj", "o_proj")}
    return W, A


def _trained(W):
    from torch import nn

    from experts4bit_qlora._vendor.experts import Experts4bit
    from experts4bit_qlora.lora import ExpertsLoRA
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList()
    for li in range(L):
        layer = nn.Module()
        layer.mlp = nn.Module()
        gu = torch.stack([torch.cat([W[(li, e, "gate_proj")], W[(li, e, "up_proj")]], 0) for e in range(E)])
        dn = torch.stack([W[(li, e, "down_proj")] for e in range(E)])
        layer.mlp.experts = ExpertsLoRA(Experts4bit.from_float(gu, dn, quant_type="nf4", blocksize=64), r=R, alpha=ALPHA)
        root.model.layers.append(layer)
    return root


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The tiny fixture's inputs: checkpoint snapshot, fixture-base, its record, both receipts, both v1 adapters."""
    from safetensors.torch import save_file

    from experts4bit_qlora.absmax_dq import compress_expert_absmax_
    tmp = tmp_path_factory.mktemp("d0s")
    W, A = _weights()
    snap = tmp / "models--org--tiny-moe" / "snapshots" / REV
    snap.mkdir(parents=True)
    tensors = {f"model.layers.{li}.mlp.experts.{e}.{p}.weight": t for (li, e, p), t in W.items()}
    tensors.update({f"model.layers.{li}.self_attn.{p}.weight": t for (li, p), t in A.items()})
    save_file(tensors, str(snap / "model.safetensors"))
    (snap / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: "model.safetensors" for k in tensors}}))
    (snap / "config.json").write_text(json.dumps({"num_experts": E, "hidden_size": H, "intermediate_size": INTER,
                                                  "num_hidden_layers": L}))
    nodq = _trained(W)
    dq = copy.deepcopy(nodq)
    assert compress_expert_absmax_(dq) == L
    base = tmp / "fixture-base"
    base.mkdir()
    stacks, payloads = {}, {}
    for li, layer in enumerate(nodq.model.layers):
        s = layer.mlp.experts.base
        for w in V2.WHICH:
            stacks[f"{li}.{w}_proj"] = getattr(s, f"{w}_proj").detach().clone()
            stacks[f"{li}.{w}_absmax"] = getattr(s, f"{w}_absmax").detach().clone()
            for x in V2.SUFFIXES:
                payloads[f"{li}.{w}_absmax{x}"] = dq.model.layers[li].mlp.experts.base._buffers[f"{w}_absmax{x}"].clone()
    save_file(stacks, str(base / "stacks.safetensors"))
    save_file(payloads, str(base / "absmax_dq.safetensors"))
    digests = {"nodq": expert_digest(nodq), "dq": expert_digest(dq)}
    (base / "fixture_base.json").write_text(json.dumps(
        {"want": digests, "got_nodq": digests["nodq"], "got_dq": digests["dq"], "nodq_ok": True, "dq_ok": True}))
    g = torch.Generator().manual_seed(5)
    lora = {}
    for li in range(L):
        lora[f"model.layers.{li}.self_attn.q_proj.lora_A"] = torch.randn(R, H, generator=g)
        lora[f"model.layers.{li}.self_attn.q_proj.lora_B"] = torch.randn(H, R, generator=g) * 0.01
        for w, (n, k) in (("gate_up", (2 * INTER, H)), ("down", (H, INTER))):
            lora[f"{STACK.format(li)}.{w}_lora_A"] = torch.randn(E, R, k, generator=g)
            lora[f"{STACK.format(li)}.{w}_lora_B"] = torch.randn(E, n, R, generator=g) * 0.05
    out = {"snap": snap, "base": base, "record": base / "fixture_base.json",
           "e4b_repo": _repo(tmp / "e4b-repo", (ROOT / "experts4bit_qlora" / "lora.py").read_text())}
    for fx in ("nodq", "dq"):
        out[f"receipt_{fx}"] = _receipt(tmp / f"receipt-{fx}.json", fx, digests[fx])
        out[f"v1_{fx}"] = _v1(tmp / f"fixture-{fx}", lora)
    return out


def _receipt(path, fx, digest, **change):
    rec = {"run_id": f"tiny-{fx}-run", "model": {"revision": REV},
           "engaged": {"setup": dict(SETUP, expert_kernel="grouped_nf4"), "commit": REV,
                       "expert_absmax": {"form": V2.FIXTURES[fx]},
                       "lora_scale_applied": {"attention": ALPHA / R, "experts": ALPHA / R},
                       "versions": {"experts4bit-qlora": "0.56.1"}, "train_fused_qkv": 0,
                       "attn_quant": {"form": "bf16", "quant_type": None, "blocksize": None, "nested": None}},
           "correctness": {"frozen_expert_bytes_unchanged": True, "frozen_expert_digest": digest}}
    for dotted, value in change.items():
        *head, last = dotted.split("__")
        node = rec
        for k in head:
            node = node[k]
        node[last] = value
    path.write_text(json.dumps(rec))
    return path


def _v1(d, tensors, setup=None):
    """A v1 adapter as Loggetta writes one: the weights plus its tokenizer files, every file in the manifest."""
    from safetensors.torch import save_file
    d.mkdir()
    save_file(tensors, str(d / "adapter.safetensors"))
    (d / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "tiny"}))
    (d / "chat_template.jinja").write_text("{{ messages }}")
    files = {n: {"sha256": hashlib.sha256((d / n).read_bytes()).hexdigest(), "bytes": (d / n).stat().st_size}
             for n in ("adapter.safetensors", "tokenizer_config.json", "chat_template.jinja")}
    man = {"schema": "backend-adapter/1", "backend": "experts4bit", "weights": "adapter.safetensors",
           "files": files,
           "resolved_model_revision": REV, "setup": dict(SETUP, **(setup or {})),
           "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()}}
    (d / "adapter_manifest.json").write_text(json.dumps(man))
    return d


def _derive(world, fx, out, **kw):
    args = dict(fixture=fx, v1_dir=str(world[f"v1_{fx}"]), receipt_path=str(world[f"receipt_{fx}"]),
                base_dir=str(world["base"]), base_record=str(world["record"]), checkpoint=str(world["snap"]),
                e4b_repo=str(world["e4b_repo"]), created_at="2026-10-24T00:00:00Z", out=str(out),
                expert_digest=expert_digest)
    args.update(kw)
    return V2.derive(**args)


# ---- the derivation ------------------------------------------------------------------------------------------------


def test_both_fixtures_derive_a_v2_the_reader_accepts_with_the_same_weights(world, tmp_path):
    from experts4bit_qlora.adapter_dir import verify_adapter_dir
    mans = {fx: _derive(world, fx, tmp_path / fx) for fx in ("nodq", "dq")}
    for fx, man in mans.items():
        v2, v1 = verify_adapter_dir(tmp_path / fx), verify_adapter_dir(world[f"v1_{fx}"])
        assert v2.manifest["schema"] == "backend-adapter/2" and v2.adapter_id == v1.adapter_id
        for name in ("adapter.safetensors", "tokenizer_config.json", "chat_template.jinja"):
            assert not os.path.islink(tmp_path / fx / name)
            assert (tmp_path / fx / name).read_bytes() == (Path(world[f"v1_{fx}"]) / name).read_bytes()
        assert man["lora_scale_applied"] == {"attention": 2.0, "experts": 2.0} and man["run_id"] == f"tiny-{fx}-run"
        d = man["derived_from"]
        assert d["limitation"].startswith("fingerprints measured after training, over the verified stored base")
        assert "2 of 16 stacks verified against the trainer digest" in d["limitation"]
        assert d["scale"] == "equal by the reader rule; the training code applies alpha/r"
        assert set(d["fixture_base_files"]) == ({"stacks.safetensors", "absmax_dq.safetensors"} if fx == "dq"
                                                else {"stacks.safetensors"})
    nodq_fp, dq_fp = (mans[fx]["fingerprints"]["experts"] for fx in ("nodq", "dq"))
    assert "fp32_source" not in nodq_fp and dq_fp["sha256"] != nodq_fp["sha256"]
    assert dq_fp["fp32_source"]["sha256"] == nodq_fp["sha256"]                 # the dq base's fp32 form IS the nodq base
    assert mans["nodq"]["fingerprints"]["attention"] == mans["dq"]["fingerprints"]["attention"]


def test_two_runs_write_byte_identical_manifests(world, tmp_path):
    for fx in ("nodq", "dq"):
        _derive(world, fx, tmp_path / f"{fx}-1")
        _derive(world, fx, tmp_path / f"{fx}-2")
        assert (tmp_path / f"{fx}-1" / "adapter_manifest.json").read_bytes() == \
            (tmp_path / f"{fx}-2" / "adapter_manifest.json").read_bytes()


@pytest.mark.parametrize("fx", ["nodq", "dq"])
def test_the_writer_and_refusal_13_take_the_derived_adapter(world, tmp_path, fx):
    nb = _need("nvme_bake_nf4")
    from experts4bit_qlora import merged_arena as MA
    from experts4bit_qlora.adapter_dir import verify_adapter_dir
    ad = tmp_path / "ad"
    _derive(world, fx, ad)
    arena = str(tmp_path / "base.arena")
    nb.bake_nf4(str(world["snap"]), arena, quantize_fn=nb.default_quantize_expert(dev="cpu"), log=lambda *a, **k: None)
    out = str(tmp_path / "merged.arena")
    block = MA.write(arena, str(ad), out, log=lambda *a: None)
    assert block["trained_base"] == V2.FIXTURES[fx].replace("fp32", "fp32-absmax")
    idx = json.load(open(out + ".index.json"))
    assert MA.merged_arena_refusals(verify_adapter_dir(ad), idx, opt_in=True) == []


# ---- every link refuses when it breaks -------------------------------------------------------------------------------


def _flip(path, key, offset=0):
    from safetensors.torch import load_file, save_file
    t = load_file(str(path))
    v = t[key].view(torch.uint8).reshape(-1)
    v[offset] ^= 0x10
    save_file(t, str(path))


def _copy_world(world, tmp_path):
    w = dict(world)
    w["base"] = tmp_path / "fixture-base"
    shutil.copytree(world["base"], w["base"])
    w["record"] = w["base"] / "fixture_base.json"
    return w


def test_a_stored_base_not_the_trainers_is_refused(world, tmp_path):
    w = _copy_world(world, tmp_path)
    _flip(w["base"] / "stacks.safetensors", "0.gate_up_proj")
    with pytest.raises(V2.Refused, match="is not the trainer's"):
        _derive(w, "nodq", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_recorded_fingerprint_that_is_not_the_bakes_is_caught_by_the_writer(world, tmp_path):
    """What backs the stated limitation (14 of 16 real stacks rest on the rebuild, not the trainer's digest): a stored
    stack that differed there would make the recorded fingerprint differ from the served bake's, and the writer refuses
    to merge into a base it cannot match. Here the recorded fingerprint carries a flipped digit."""
    nb = _need("nvme_bake_nf4")
    from experts4bit_qlora import merged_arena as MA
    ad = tmp_path / "ad"
    _derive(world, "nodq", ad)
    man = json.loads((ad / "adapter_manifest.json").read_text())
    fp = man["fingerprints"]["experts"]
    fp["sha256"] = ("0" if fp["sha256"][0] != "0" else "1") + fp["sha256"][1:]
    (ad / "adapter_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    arena = str(tmp_path / "base.arena")
    nb.bake_nf4(str(world["snap"]), arena, quantize_fn=nb.default_quantize_expert(dev="cpu"), log=lambda *a, **k: None)
    with pytest.raises(MA.MergeRefused, match="not the base this adapter was trained on"):
        MA.write(arena, str(ad), str(tmp_path / "m.arena"), log=lambda *a: None)


@pytest.mark.parametrize("fx, change, needle", [
    ("nodq", {"engaged__lora_scale_applied": {"attention": 2.0, "experts": 4.0}}, "is not the recorded alpha / r"),
    ("dq", {"engaged__expert_absmax": {"form": "fp32"}}, "expert_absmax form is 'fp32'"),
    ("nodq", {"engaged__commit": "d" * 40}, "is not the training receipt's"),
    ("nodq", {"engaged__setup": dict(SETUP, use_rslora=True)}, "carries a scaling option"),
    ("nodq", {"correctness__frozen_expert_bytes_unchanged": False}, "no unchanged frozen-expert digest"),
    ("nodq", {"engaged__setup": dict(SETUP, alpha=8)}, "setup alpha"),
])
def test_a_training_record_that_does_not_match_is_refused(world, tmp_path, fx, change, needle):
    w = dict(world)
    rec = json.loads(Path(world[f"receipt_{fx}"]).read_text())
    w[f"receipt_{fx}"] = _receipt(tmp_path / "receipt.json", fx, rec["correctness"]["frozen_expert_digest"], **change)
    with pytest.raises(V2.Refused, match=needle):
        _derive(w, fx, tmp_path / "out")


def test_a_double_quantized_fixture_needs_its_stored_payload(world, tmp_path):
    w = _copy_world(world, tmp_path)
    os.remove(w["base"] / "absmax_dq.safetensors")
    with pytest.raises(V2.Refused, match="needs fixture-base's absmax_dq.safetensors"):
        _derive(w, "dq", tmp_path / "out")


def test_a_dq_fingerprint_without_its_fp32_source_is_refused(world, tmp_path, monkeypatch):
    from experts4bit_qlora import absmax_dq
    monkeypatch.setattr(absmax_dq, "absmax_source_sha256", lambda mod, which: None)
    with pytest.raises(V2.Refused, match="carries no fp32_source"):
        _derive(world, "dq", tmp_path / "out")


def test_a_record_that_does_not_verify_the_base_is_refused(world, tmp_path):
    w = _copy_world(world, tmp_path)
    rec = json.loads(Path(w["record"]).read_text())
    rec["nodq_ok"] = False
    Path(w["record"]).write_text(json.dumps(rec))
    with pytest.raises(V2.Refused, match="fixture_base.json does not record"):
        _derive(w, "nodq", tmp_path / "out")


@pytest.mark.parametrize("text, needle", [
    ("class ExpertsLoRA:\n    self.scaling = alpha / r\nclass LoRALinear:\n    self.use_rslora = True\n"
     "    self.scaling = alpha / r\n", "rslora"),
    ("class ExpertsLoRA:\n    self.scaling = alpha / math.sqrt(r)\nclass LoRALinear:\n    self.scaling = alpha / r\n",
     "sqrt"),
    ("class ExpertsLoRA:\n    self.scaling = alpha / r\nclass LoRALinear:\n    self.scaling = alpha / r\n"
     "    self.scaling = 1.0\n", "assigns self.scaling"),
])
def test_a_training_lora_py_that_does_not_apply_alpha_over_r_is_refused(world, tmp_path, text, needle):
    with pytest.raises(V2.Refused, match=needle):
        _derive(world, "nodq", tmp_path / "out", e4b_repo=str(_repo(tmp_path / "repo", text)))


def test_the_lora_py_read_is_the_training_versions_tag_and_is_recorded(world, tmp_path):
    man = _derive(world, "nodq", tmp_path / "out")
    rec = man["derived_from"]["lora_py"]
    repo = str(world["e4b_repo"])
    commit = subprocess.run(["git", "-C", repo, "rev-parse", "v0.56.1^{commit}"], capture_output=True, text=True).stdout.strip()
    blob = subprocess.run(["git", "-C", repo, "rev-parse", f"{commit}:experts4bit_qlora/lora.py"], capture_output=True,
                          text=True).stdout.strip()
    assert rec == {"version": "0.56.1", "tag": "v0.56.1", "commit": commit, "blob": blob,
                   "sha256": hashlib.sha256((ROOT / "experts4bit_qlora" / "lora.py").read_bytes()).hexdigest()}


def test_a_training_version_the_repository_has_no_tag_for_is_refused(world, tmp_path):
    w = dict(world)
    rec = json.loads(Path(world["receipt_nodq"]).read_text())
    w["receipt_nodq"] = _receipt(tmp_path / "receipt.json", "nodq", rec["correctness"]["frozen_expert_digest"],
                                 engaged__versions={"experts4bit-qlora": "9.9.9"})
    with pytest.raises(V2.Refused, match="refs/tags/v9.9.9"):
        _derive(w, "nodq", tmp_path / "out")


def test_a_tag_whose_lora_py_is_not_the_one_read_fails_its_check(world, tmp_path):
    """The bytes are the tag's, never a file passed beside it: a tag whose lora.py carries rslora is refused even though
    main's lora.py would pass."""
    bad = (ROOT / "experts4bit_qlora" / "lora.py").read_text() + "\nuse_rslora = False\n"
    with pytest.raises(V2.Refused, match="rslora"):
        _derive(world, "nodq", tmp_path / "out", e4b_repo=str(_repo(tmp_path / "repo", bad)))


@pytest.mark.parametrize("which, field, bad, needle", [
    ("gate_up", "_s", lambda t: t[:-1], "gate_up_absmax_s is torch.float32 x"),
    ("down", "_s", lambda t: t[:-1], "down_absmax_s is torch.float32 x"),
    ("gate_up", "_q", lambda t: t.to(torch.int16), "_absmax_q is torch.int16"),
    ("gate_up", "_off", lambda t: t.repeat(2), "_absmax_off is torch.float32 x 2"),
    ("gate_up", "_code", lambda t: t[:128], "_absmax_code is torch.float32 x 128"),
])
def test_installing_a_payload_of_the_wrong_shape_is_refused_before_anything_moves(which, field, bad, needle):
    from experts4bit_qlora.absmax_dq import compress_expert_absmax_, install_compressed_absmax_
    W, _A = _weights()
    a, b = _trained(W), _trained(W)
    compress_expert_absmax_(a)
    sa = a.model.layers[0].mlp.experts.base
    payloads = {w: {x: sa._buffers[f"{w}_absmax{x}"] for x in V2.SUFFIXES} for w in V2.WHICH}
    payloads[which][field] = bad(payloads[which][field])
    with pytest.raises(ValueError, match=needle):
        install_compressed_absmax_(b.model.layers[0].mlp.experts.base, payloads)
    sb = b.model.layers[0].mlp.experts.base
    assert all(f"{w}_absmax" in sb._buffers for w in V2.WHICH)                   # a bad down payload moves no gate_up
    assert not any(k.endswith("_absmax_q") for k in sb._buffers)


def test_the_real_training_code_and_main_pass_the_scaling_check():
    V2.check_scaling_code((ROOT / "experts4bit_qlora" / "lora.py").read_text())


def test_the_derived_directory_is_always_new(world, tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(V2.Refused, match="exists"):
        _derive(world, "nodq", tmp_path / "out")


def test_the_v1_fixture_is_only_read(world, tmp_path):
    before = {p.name: p.read_bytes() for p in Path(world["v1_nodq"]).iterdir()}
    _derive(world, "nodq", tmp_path / "out")
    assert {p.name: p.read_bytes() for p in Path(world["v1_nodq"]).iterdir()} == before


def test_compression_installs_through_the_shared_routine_unchanged(world):
    """compress_expert_absmax_ now installs through install_compressed_absmax_; reinstalling a stack's own payload
    on a fresh fp32 copy gives the same buffers, mark and fingerprint."""
    from experts4bit_qlora import adapter_schema as S
    from experts4bit_qlora.absmax_dq import compress_expert_absmax_, install_compressed_absmax_
    W, _A = _weights()
    a, b = _trained(W), _trained(W)
    compress_expert_absmax_(a)
    for la, lb in zip(a.model.layers, b.model.layers):
        sa = la.mlp.experts.base
        install_compressed_absmax_(lb.mlp.experts.base, {w: {x: sa._buffers[f"{w}_absmax{x}"] for x in V2.SUFFIXES}
                                                         for w in V2.WHICH})
    fa, fb = S.quant_fingerprint(a, "experts"), S.quant_fingerprint(b, "experts")
    assert fa["sha256"] == fb["sha256"] and fa["fp32_source"] == fb["fp32_source"]
    assert expert_digest(a) == expert_digest(b)
