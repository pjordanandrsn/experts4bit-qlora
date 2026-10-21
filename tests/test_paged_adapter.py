"""PR-A: one trained adapter on the paged server (experts4bit_qlora/adapter_dir.py, docs/ADAPTER_PORTABILITY.md).

CPU only. Adapter directories are written here exactly as Loggetta's ``save_adapter`` writes them (backend-adapter/1:
``adapter.safetensors`` + ``adapter_manifest.json`` with per-file sha256 and per-tensor shape/dtype); the model is a
tiny real Qwen3-MoE from transformers, so the merge, the name mapping and ``fuse_qkv`` run on the real modules.
"""
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
tr = pytest.importorskip("transformers")

from experts4bit_qlora import adapter_dir as AD  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REV = "a" * 40
LAYER = "model.layers.0.self_attn"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
ARENA = {"snapshot": f"/cache/models--x--y/snapshots/{REV}", "quantizer": {"quant_type": "nf4", "blocksize": 64}}
NO_LEVERS = {}


def _model():
    cfg = tr.Qwen3MoeConfig(hidden_size=64, intermediate_size=64, moe_intermediate_size=32, num_experts=4,
                            num_experts_per_tok=2, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, vocab_size=128, max_position_embeddings=64)
    torch.manual_seed(0)
    return tr.Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).eval()


def _lora(model, r=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for p in PROJ:
        lin = model.get_submodule(f"{LAYER}.{p}")
        out[f"{LAYER}.{p}.lora_A"] = torch.randn(r, lin.in_features, generator=g).to(torch.bfloat16)
        out[f"{LAYER}.{p}.lora_B"] = torch.randn(lin.out_features, r, generator=g).to(torch.bfloat16) * 0.1
    return out


def _write(tmp_path, tensors, *, setup=None, revision=REV, schema="backend-adapter/1", name="ad"):
    from safetensors.torch import save_file

    d = tmp_path / name
    d.mkdir()
    save_file(tensors, str(d / "adapter.safetensors"), metadata={"format": schema, "backend": "experts4bit"})
    files = {"adapter.safetensors": {"sha256": hashlib.sha256((d / "adapter.safetensors").read_bytes()).hexdigest(),
                                     "bytes": (d / "adapter.safetensors").stat().st_size}}
    st = {"quant_type": "nf4", "blocksize": 64, "r": 2, "alpha": 4, "attn_4bit": False, "train_experts": False,
          "train_attention": True}
    st.update(setup or {})
    man = {"schema": schema, "backend": "experts4bit", "weights": "adapter.safetensors", "files": files,
           "resolved_model_revision": revision, "setup": st,
           "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()}}
    (d / "adapter_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return d


def _refusals(ad, **kw):
    args = dict(served_revision=REV, requested_revision=None, arena_index=ARENA, levers=NO_LEVERS,
                accept_unverified_base=frozenset({"experts"}), served_model="x/y")
    args.update(kw)
    return AD.servable_refusals(ad, **args)


def _codes(refusals):
    return [c for c, _ in refusals]


# ---------------------------------------------------------------------------------------------------------- verifier

def test_a_good_directory_verifies_and_maps_each_projection(tmp_path):
    m = _model()
    d = _write(tmp_path, _lora(m))
    ad = AD.verify_adapter_dir(d)
    assert set(ad.attention) == {f"{LAYER}.{p}" for p in PROJ}
    assert ad.scale == 2.0 and ad.revision == REV                             # alpha 4 / r 2, from the manifest
    assert ad.adapter_id == hashlib.sha256((d / "adapter.safetensors").read_bytes()).hexdigest()
    assert ad.expert_tensors == () and ad.other_tensors == ()
    assert _refusals(ad) == []


def _tamper(tmp_path, how):
    m = _model()
    t = _lora(m)
    if how == "extra":
        d = _write(tmp_path, t)
        man = json.loads((d / "adapter_manifest.json").read_text())
        man["tensors"]["model.layers.0.self_attn.x.lora_A"] = {"shape": [2, 64], "dtype": "torch.bfloat16"}
        (d / "adapter_manifest.json").write_text(json.dumps(man))
        return d
    if how == "nan":
        t[f"{LAYER}.q_proj.lora_B"][0, 0] = float("nan")
    if how == "lone_a":
        del t[f"{LAYER}.q_proj.lora_B"]
    d = _write(tmp_path, t, schema="backend-adapter/0" if how == "schema" else "backend-adapter/1",
               setup={"r": None} if how == "no_scale" else None)
    if how == "byte":
        raw = bytearray((d / "adapter.safetensors").read_bytes())
        raw[-1] ^= 0x01
        (d / "adapter.safetensors").write_bytes(bytes(raw))
    if how in ("shape", "dtype"):
        man = json.loads((d / "adapter_manifest.json").read_text())
        rec = man["tensors"][f"{LAYER}.q_proj.lora_A"]
        rec["shape" if how == "shape" else "dtype"] = [3, 64] if how == "shape" else "torch.float32"
        (d / "adapter_manifest.json").write_text(json.dumps(man))
    if how == "traversal":
        man = json.loads((d / "adapter_manifest.json").read_text())
        man["files"]["../evil"] = {"sha256": "0" * 64, "bytes": 1}
        (d / "adapter_manifest.json").write_text(json.dumps(man))
    return d


@pytest.mark.parametrize("how,needle", [
    ("byte", "sha256 does not match"), ("extra", "differ from the manifest"), ("nan", "non-finite"),
    ("schema", "schema"), ("shape", "shape/dtype"), ("dtype", "shape/dtype"), ("traversal", "unsafe file path"),
    ("no_scale", "alpha/r cannot be read"), ("lone_a", "must come together")])
def test_the_verifier_refuses_each_fault_by_name(tmp_path, how, needle):
    with pytest.raises(AD.AdapterRefused) as e:
        AD.verify_adapter_dir(_tamper(tmp_path, how))
    assert e.value.code == 7 and needle in str(e.value), str(e.value)


def test_the_verifier_refuses_a_symlinked_weights_file(tmp_path):
    m = _model()
    d = _write(tmp_path, _lora(m))
    real = tmp_path / "elsewhere.safetensors"
    (d / "adapter.safetensors").rename(real)
    try:
        (d / "adapter.safetensors").symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("this platform cannot make a symlink here")
    with pytest.raises(AD.AdapterRefused) as e:
        AD.verify_adapter_dir(d)
    assert e.value.code == 7 and "unsafe file path" in str(e.value)


# -------------------------------------------------------------------------------------------------------------- merge

def test_the_merge_is_w_plus_scale_b_a_in_fp32_cast_once(tmp_path):
    m = _model()
    t = _lora(m)
    before = {p: m.get_submodule(f"{LAYER}.{p}").weight.detach().clone() for p in PROJ}
    ad = AD.verify_adapter_dir(_write(tmp_path, t, setup={"alpha": 3}))            # alpha 3, r 2: scale 1.5, not 2
    rep = AD.merge_attention_lora(m, ad)
    assert rep.merged == 4 and rep.scale == 1.5
    for p in PROJ:
        a, b = t[f"{LAYER}.{p}.lora_A"].float(), t[f"{LAYER}.{p}.lora_B"].float()
        exact = before[p].float() + 1.5 * (b @ a)
        got = m.get_submodule(f"{LAYER}.{p}").weight.detach()
        assert torch.equal(got, exact.to(torch.bfloat16)), p
        assert rep.projections[f"{LAYER}.{p}"]["max_err"] == float((got.float() - exact).abs().max())
    assert 0 < rep.to_dict()["max_err"] < rep.to_dict()["max_abs_delta"]


def test_the_merge_checks_every_target_before_touching_any(tmp_path):
    m = _model()
    t = _lora(m)
    t["model.layers.0.self_attn.nope_proj.lora_A"] = t.pop(f"{LAYER}.o_proj.lora_A")    # not an attention name
    t["model.layers.0.self_attn.nope_proj.lora_B"] = t.pop(f"{LAYER}.o_proj.lora_B")
    t[f"{LAYER}.o_proj.lora_A"] = torch.zeros(2, 64, dtype=torch.bfloat16)
    t[f"{LAYER}.o_proj.lora_B"] = torch.zeros(64, 2, dtype=torch.bfloat16)
    ad = AD.verify_adapter_dir(_write(tmp_path, t))
    assert _codes(_refusals(ad)) == [6]                                      # an unknown target module is refused
    m2 = _model()
    lin = m2.get_submodule(f"{LAYER}.k_proj")
    m2.get_submodule(LAYER).k_proj = type("NotPlain", (torch.nn.Linear,), {})(lin.in_features, lin.out_features,
                                                                              bias=False, dtype=torch.bfloat16)
    q_before = m2.get_submodule(f"{LAYER}.q_proj").weight.detach().clone()
    good = AD.verify_adapter_dir(_write(tmp_path, _lora(m2), name="ad2"))
    with pytest.raises(AD.AdapterRefused) as e:
        AD.merge_attention_lora(m2, good)
    assert e.value.code == 7 and "not a plain nn.Linear" in str(e.value)
    assert torch.equal(m2.get_submodule(f"{LAYER}.q_proj").weight, q_before)   # nothing merged


def test_fuse_qkv_after_the_merge_carries_the_merged_weights(tmp_path):
    from experts4bit_qlora.engines.qkv_fuse import fuse_qkv

    m = _model()
    ad = AD.verify_adapter_dir(_write(tmp_path, _lora(m)))
    AD.merge_attention_lora(m, ad)
    merged = [m.get_submodule(f"{LAYER}.{p}").weight.detach().clone() for p in ("q_proj", "k_proj", "v_proj")]
    assert fuse_qkv(m, fold_modes={"E4B_FUSE_T1_GLUE": "0", "E4B_FUSE_T1_GLUE_R2": "0",
                                   "E4B_FUSE_ROUTER_EPI": "0"}) == 1
    assert torch.equal(m.get_submodule(LAYER).qkv_proj.weight, torch.cat(merged))


def test_build_engine_merges_after_the_load_and_before_every_lever_fusion_and_route():
    src = (ROOT / "experts4bit_qlora" / "serve_paged.py").read_text(encoding="utf-8")
    body = src[src.index("def build_engine("):]
    order = [body.index(x) for x in ("adapter = verify_adapter_dir(cfg.adapter)", "load_moe_4bit_streaming(",
                                     "report = merge_attention_lora(model, adapter)", "enable_hybrid_tier(",
                                     "levers = _apply_levers(", "fusions = _apply_fusions(",
                                     "dense_gemv = apply_dense_gemv_bw(", "int8_head = apply_int8_head_bw_resolved(")]
    assert order == sorted(order)


# ---------------------------------------------------------------------------------------------------------- refusals

def _ad(tmp_path, **kw):
    m = _model()
    t = _lora(m)
    if kw.pop("other", False):
        t["model.layers.0.mlp.gate.lora_A"] = torch.zeros(2, 64, dtype=torch.bfloat16)   # the router: never adapted
    if kw.pop("experts", False):
        t["model.layers.0.mlp.experts.gate_up_lora_A"] = torch.zeros(4, 2, 64, dtype=torch.bfloat16)
        t["model.layers.0.mlp.experts.gate_up_lora_B"] = torch.full((4, 64, 2), 0.01, dtype=torch.bfloat16)
    return AD.verify_adapter_dir(_write(tmp_path, t, **kw))


def test_each_refusal_names_its_reason(tmp_path):
    assert _codes(_refusals(_ad(tmp_path, name="e", experts=True))) == [1]
    assert _codes(_refusals(_ad(tmp_path, name="o", other=True))) == [6]
    assert _codes(_refusals(_ad(tmp_path, name="r", revision=None))) == [2]
    ad = _ad(tmp_path, name="ok")
    assert _codes(_refusals(ad, served_revision="b" * 40)) == [2]
    assert _codes(_refusals(ad, served_revision=None)) == [2]                 # an unknown served commit is not a match
    assert _refusals(ad, served_revision=None, check_served_revision=False) == []
    assert _codes(_refusals(ad, requested_revision="c" * 40)) == [2]
    local = {**ARENA, "snapshot": "/local/dir"}
    assert _codes(_refusals(ad, arena_index=local)) == [3]
    assert _refusals(ad, arena_index=local, allow_unverified_arena=True) == []
    assert _codes(_refusals(ad, arena_index={**ARENA, "snapshot": "/c/models--x--y/snapshots/" + "d" * 40})) == [3]
    assert _codes(_refusals(ad, served_model="x/other")) == [3]                # the repo id, not just the revision
    assert _refusals(ad, served_model="/local/snapshot/dir") == []            # a local path carries no repo id
    assert _codes(_refusals(ad, arena_index={**ARENA, "quantizer": {"quant_type": "fp4", "blocksize": 64}})) == [4]
    assert _codes(_refusals(ad, arena_index={"snapshot": ARENA["snapshot"]})) == [4]   # no quantizer: fails closed
    assert _codes(_refusals(ad, arena_index={**ARENA, "quantizer": {"quant_type": "nf4"}})) == [4]
    assert _codes(_refusals(ad, spec="eagle3")) == [10]
    assert _codes(_refusals(ad, levers={"E4B_SERVE_ATTN_INT8": True})) == [5]
    assert _codes(_refusals(ad, levers={"E4B_SERVE_EXP_INT4": True})) == [5]  # the experts around the adapter, too
    assert _codes(_refusals(_ad(tmp_path, name="q", setup={"attn_4bit": True}))) == [8]
    nine = _refusals(ad, accept_unverified_base=frozenset())
    assert _codes(nine) == [9] and "ADAPTER_PORTABILITY.md F1" in nine[0][1]
    assert "E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE=experts" in nine[0][1]


def test_a_zero_expert_b_is_not_a_live_expert_adapter(tmp_path):
    m = _model()
    t = _lora(m)
    t["model.layers.0.mlp.experts.gate_up_lora_A"] = torch.ones(4, 2, 64, dtype=torch.bfloat16)
    t["model.layers.0.mlp.experts.gate_up_lora_B"] = torch.zeros(4, 64, 2, dtype=torch.bfloat16)
    assert _refusals(AD.verify_adapter_dir(_write(tmp_path, t))) == []


def test_the_unverified_base_opt_in_names_its_scope():
    assert AD.parse_accept_unverified_base("experts") == frozenset({"experts"})
    assert AD.parse_accept_unverified_base("") == frozenset()
    for bad in ("1", "true", "experts,attention", "yes"):
        with pytest.raises(ValueError, match="unknown scope"):
            AD.parse_accept_unverified_base(bad)


def test_base_match_states_each_part(tmp_path):
    ad = _ad(tmp_path)
    bm = AD.base_match(ad)
    assert bm["attention"] == "exact by construction (up to the measured merge rounding)"
    assert bm["experts"].startswith("unverified") and "F1" in bm["experts"]
    assert AD.base_match(ad, [(8, "x")])["attention"] == "refused"


def test_the_arena_revision_is_read_only_from_a_snapshot_commit():
    assert AD.arena_source_revision(ARENA) == REV
    assert AD.arena_source_revision({"snapshot": "/models/qwen"}) is None
    assert AD.arena_source_revision({"snapshot": f"/x/snapshots/{REV}\n"}) is None   # fullmatch: no trailing newline
    assert AD.arena_source_revision({}) is None
    assert AD.arena_source(ARENA) == ("x/y", REV)
    assert AD.arena_source({"snapshot": f"C:\\hf\\models--Qwen--Qwen3-30B-A3B\\snapshots\\{REV}"}) == \
        ("Qwen/Qwen3-30B-A3B", REV)


# ------------------------------------------------------------------------------------------------- config + identity

def test_the_config_refuses_opt_ins_without_an_adapter_and_a_loose_flag(monkeypatch):
    from experts4bit_qlora.serve_paged import PagedServeConfig, _zero_one

    with pytest.raises(ValueError, match="E4B_PAGED_ADAPTER is not"):
        PagedServeConfig(model="m", adapter_accept_unverified_base=frozenset({"experts"})).validate()
    with pytest.raises(ValueError, match="expected '0' or '1'"):
        _zero_one("E4B_PAGED_ADAPTER_ALLOW_UNVERIFIED_ARENA", "yes")
    monkeypatch.setenv("E4B_PAGED_ADAPTER", "/some/dir")
    monkeypatch.setenv("E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE", "1")
    with pytest.raises(ValueError, match="unknown scope"):
        PagedServeConfig.from_env()


def test_while_the_adapter_loads_the_base_name_is_not_listed():
    from experts4bit_qlora.serve_paged import PagedServeConfig

    loading = PagedServeConfig(model="tiny/moe", adapter="/ad", served_names=("alias",))
    assert loading.model_names == ("alias",)
    built = PagedServeConfig(model="tiny/moe", adapter="/ad", adapter_id="e" * 64)
    assert built.model_names == ("tiny/moe+" + "e" * 12,)
    assert PagedServeConfig(model="tiny/moe").model_names == ("tiny/moe",)


#: an environment read in any spelling the two functions use: env("X"), os.environ.get("X"), os.environ["X"], getenv
_ENV_READ = re.compile(r'(?:\benv|environ\.get|getenv)\(\s*"([A-Z][A-Z0-9_]+)"|environ\[\s*"([A-Z][A-Z0-9_]+)"')


def _function_body(path, name):
    """The source of top-level or nested ``def name(`` up to the next line indented no deeper than its ``def``."""
    lines = path.read_text(encoding="utf-8").split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith(f"def {name}("))
    base = len(lines[start]) - len(lines[start].lstrip())
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip()
                and len(lines[i]) - len(lines[i].lstrip()) <= base), len(lines))
    return "\n".join(lines[start:end])


def _env_reads(path, name):
    return {a or b for a, b in _ENV_READ.findall(_function_body(path, name))}


def test_the_census_reader_sees_every_spelling():
    src = 'env("A_B")  os.environ.get("C_D", "0")  os.environ["E_F"]  os.getenv("G_H")  myenv("NOT_ME")'
    assert {a or b for a, b in _ENV_READ.findall(src)} == {"A_B", "C_D", "E_F", "G_H"}


def test_every_lever_is_classified_refused_or_allowed_with_its_reason():
    """The census: every env name _apply_levers and the lane hook it mirrors read, WHATEVER ITS PREFIX, plus the int8
    head, sits in exactly one table with a reason. A refused lever re-quantizes an adapted module (i) or depends on
    base-model calibration (ii); a config name only configures refused levers."""
    from experts4bit_qlora import serve_paged as sp

    read = (_env_reads(ROOT / "experts4bit_qlora" / "serve_paged.py", "_apply_levers")
            | _env_reads(ROOT / "bench" / "p42" / "hook" / "usercustomize.py", "_apply_lanes")
            | {"E4B_INT8_HEAD_BW"})
    refused, allowed, config = set(sp.ADAPTER_REFUSED_LEVERS), set(sp.ADAPTER_ALLOWED_LEVERS), set(sp.ADAPTER_LEVER_CONFIG)
    assert read == refused | allowed | config, ("unclassified", read - refused - allowed - config,
                                                "stale", (refused | allowed | config) - read)
    assert not (refused & allowed) and not (refused & config) and not (allowed & config)
    assert all(levers and set(levers) <= refused for levers in sp.ADAPTER_LEVER_CONFIG.values())
    assert all(("(i" in why or "(ii)" in why) for why in sp.ADAPTER_REFUSED_LEVERS.values())
    assert all("CALIB" not in k or "(ii)" in why for k, why in sp.ADAPTER_REFUSED_LEVERS.items())
    assert all("weight-only" in why and "no activation statistics" in why for why in sp.ADAPTER_ALLOWED_LEVERS.values())


def test_health_and_models_carry_the_adapter_and_the_base_name_is_not_served():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import test_serve_paged as tsp
    finally:
        sys.path.remove(str(Path(__file__).parent))
    from fastapi.testclient import TestClient

    from experts4bit_qlora.serve_paged import PagedEngine, PagedServeConfig, create_app

    aid = "f" * 64
    cfg = PagedServeConfig(model="tiny/moe", max_seqs=4, max_tokens_per_seq=64, chunk_tokens=8, adapter="/ad",
                           adapter_id=aid, adapter_accept_unverified_base=frozenset({"experts"}))
    parts = tsp._parts(tsp.ScriptedRunner(), max_seqs=4, chunk=8)
    parts.info["adapter"] = {"adapter_id": aid, "manifest_sha256": "m" * 64, "base_revision": REV,
                             "base_match": {"attention": "exact", "experts": "unverified (F1)"}, "merge": {"merged": 4}}
    with TestClient(create_app(cfg, engine=PagedEngine(cfg, parts))) as c:
        h = c.get("/health").json()
        assert h["adapter"]["adapter_id"] == aid and h["adapter"]["base_match"]["experts"].startswith("unverified")
        assert "adapter" not in h["levers"]
        assert h["served_model_names"] == [f"tiny/moe+{aid[:12]}"]
        m = c.get("/v1/models").json()["data"]
        assert [d["id"] for d in m] == [f"tiny/moe+{aid[:12]}"]                  # model_id "tiny/moe" is not re-added
        assert m[0]["adapter"]["base_revision"] == REV and m[0]["adapter"]["manifest_sha256"] == "m" * 64
        r = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": "the cat", "max_tokens": 2})
        assert r.status_code == 404                                           # the base is not served in its place


# ------------------------------------------------------------------------------------------------------------ mutants

#: each refusal, reverted: the suite above must notice (run against a mutated copy of adapter_dir.py)
MUTANTS = [
    ('    if live:\n', "    if False:\n"),                                                    # refusal 1
    ("    elif served_revision != ad.revision:\n", "    elif False:\n"),                         # refusal 2
    ("    if arena_rev is None:\n        if not allow_unverified_arena:\n",
     "    if arena_rev is None:\n        if False:\n"),                                           # refusal 3
    ("            if key in ad.setup and q[key] != ad.setup[key]:\n",
     "            if False:\n"),                                                                  # refusal 4
    ("        out.append((4, \"cannot verify the arena's quantization recipe",
     "        print((4, \"cannot verify the arena's quantization recipe"),                           # 4 fails closed
    ("        if served_model and \"/\" in served_model and not os.path.isabs(served_model) and arena_repo != served_model:\n",
     "        if False:\n"),                                                                     # 3: the repo id
    ('    if (spec or "off") != "off":\n', "    if False:\n"),                                 # refusal 10
    ("    if on:\n", "    if False:\n"),                                                        # refusal 5
    ("    if ad.other_tensors:\n", "    if False:\n"),                                          # refusal 6
    ('    if ad.setup.get("attn_4bit") and ad.attention:\n', "    if False:\n"),                # refusal 8
    ('    if ad.manifest.get("schema") == SCHEMA_V1 and "experts" not in accept_unverified_base:\n',
     "    if False:\n"),                                                                          # refusal 9
    ("            delta = ad.scale * (b.to(w.device, torch.float32) @ a.to(w.device, torch.float32))\n",
     "            delta = (b.to(w.device, torch.float32) @ a.to(w.device, torch.float32))\n"),  # the scale dropped
]


@pytest.mark.parametrize("old,new", MUTANTS)
def test_each_mutant_is_caught(tmp_path, monkeypatch, old, new):
    src = (ROOT / "experts4bit_qlora" / "adapter_dir.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    path = tmp_path / "adapter_dir_mutant.py"
    path.write_bytes(src.replace(old, new).encode("utf-8"))
    spec = importlib.util.spec_from_file_location("experts4bit_qlora.adapter_dir_mutant", path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)                       # dataclasses read the module while defining
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys.modules[__name__], "AD", mod)
    caught = 0
    for i, test in enumerate((test_each_refusal_names_its_reason, test_the_merge_is_w_plus_scale_b_a_in_fp32_cast_once)):
        sub = tmp_path / f"run{i}"
        sub.mkdir()
        try:
            test(sub)
        except AssertionError:                                               # only a failed check counts as caught
            caught += 1
    assert caught >= 1, f"mutant survived: {new!r}"


def test_no_name_match_uses_an_anchor_only_regex():
    src = (ROOT / "experts4bit_qlora" / "adapter_dir.py").read_text(encoding="utf-8")
    assert not re.search(r"\.match\(", src)                    # validation is fullmatch: '$' also matches before '\n'
