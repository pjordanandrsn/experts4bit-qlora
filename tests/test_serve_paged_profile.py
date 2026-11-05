# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""serve_paged's ``E4B_PAGED_PROFILE``: the placement solver's measured routing mass, refused when it does not fit the
model. Unset, the solver keeps its uniform-assumed mass, as before."""
import hashlib
import inspect
import json

import pytest

from experts4bit_qlora import serve_paged as sp
from experts4bit_qlora.engines import placement as pl

CALIB = {
    "cpu_bench": {"scatter_best": {"gbs": 200.0}},
    "gpu_bench": {"devices": [{"b_vram_triad_gbs": 1600.0}]},
}
L, E, BPE = 4, 16, 10 * 1024 * 1024
HOT = (5, 9, 13)                      # not the lowest ids, so a uniform (tie-broken) placement does not pick them


def _write(tmp_path, rows, name="prof.jsonl"):
    p = tmp_path / name
    p.write_bytes(("\n".join(json.dumps(r) for r in rows) + "\n").encode())
    return p


def _rows(layers=L, experts=E, hot=HOT):
    return ([{"row": "meta", "phase": "fit"}]
            + [{"row": "expert", "layer_id": layer, "expert_id": e, "tokens_routed": 1000 if e in hot else 1}
               for layer in range(layers) for e in range(experts)])


def _solve(profile_path, vram_slots=len(HOT) * L):
    return pl.solve_placement(n_layers=L, n_experts=E, bytes_per_expert=BPE, vram_budget_bytes=vram_slots * BPE,
                              dram_budget_bytes=24 * BPE, calibration=CALIB, profile_path=profile_path)


def test_unset_is_none():
    assert sp.placement_profile("", L, E) is None


def test_a_fitting_profile_returns_its_path(tmp_path):
    p = _write(tmp_path, _rows())
    assert sp.placement_profile(str(p), L, E) == str(p)


def test_refuses_a_profile_with_no_routed_tokens(tmp_path):
    zero = [dict(r, tokens_routed=0) if r["row"] == "expert" else r for r in _rows()]
    for rows in (zero, [{"row": "meta"}]):
        with pytest.raises(ValueError, match="no routed tokens"):
            sp.placement_profile(str(_write(tmp_path, rows)), L, E)


@pytest.mark.parametrize("rows, first", [
    (_rows(layers=L + 1), (L, 0)),                    # a deeper model's profile
    (_rows(experts=E + 4), (0, E)),                   # a wider model's profile
    (_rows() + [{"row": "expert", "layer_id": -1, "expert_id": 0, "tokens_routed": 3}], (-1, 0)),
])
def test_refuses_a_profile_from_another_model(tmp_path, rows, first):
    with pytest.raises(ValueError, match=f"outside this model's {L} layers x {E} experts") as err:
        sp.placement_profile(str(_write(tmp_path, rows)), L, E)
    assert f"first {first}" in str(err.value)


def test_the_measured_profile_moves_the_routed_mass_into_vram(tmp_path):
    """The solver balances GPU and CPU completion times (it may keep a hot expert on the CPU tier), so the property is
    the share of the REAL routed tokens its VRAM tier holds: high under the measured profile, near nothing under the
    uniform-assumed fallback, which breaks its ties away from these hot ids."""
    rows = _rows()
    p = _write(tmp_path, rows)
    m = _solve(sp.placement_profile(str(p), L, E))
    assert m["profile"] == {"kind": "measured", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    u = _solve(sp.placement_profile("", L, E))
    assert u["profile"] == {"kind": "uniform-assumed", "sha256": None}
    tokens = {(r["layer_id"], r["expert_id"]): r["tokens_routed"] for r in rows if r["row"] == "expert"}

    def vram_share(man):
        return sum(tokens[tuple(x)] for x in man["tiers"]["vram"]) / sum(tokens.values())

    assert vram_share(m) > 0.8 and vram_share(u) < 0.2


def test_from_env_reads_the_knob(monkeypatch):
    monkeypatch.setenv("E4B_PAGED_PROFILE", "/fit/expert_profile.jsonl")
    assert sp.PagedServeConfig.from_env().profile == "/fit/expert_profile.jsonl"
    monkeypatch.delenv("E4B_PAGED_PROFILE")
    assert sp.PagedServeConfig.from_env().profile == ""


def test_build_engine_hands_the_profile_to_the_solver_and_reports_it():
    """build_engine needs a GPU; its source must pass the checked profile on, check its path with the other inputs
    before any weight is read, and record the solver's profile record in /health."""
    src = inspect.getsource(sp.build_engine)
    assert src.count("profile_path=placement_profile(cfg.profile, L, E)") == 1
    assert "profile_path=None" not in src
    checked = "(cfg.arena, cfg.arena + \".index.json\", cfg.calib) + ((cfg.profile,) if cfg.profile else ())"
    assert checked in src and src.index(checked) < src.index("profile_path=placement_profile(")
    assert 'info["placement_profile"] = placement_profile_record' in src
