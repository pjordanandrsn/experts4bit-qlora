# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""The #344 load instrumentation: staged synchronisation and the shard-read diagnosis.

#344 sat open for three weeks on an error that said only ``CUDA error: invalid argument``.
Two hosts produced it at two unrelated points in the same load, a fix was merged and
reverted the same day against a threshold the model could never reach, and every reading
of the traceback was a reading of a site CUDA does not promise is the faulting one.

What was missing was never a kernel. It was that the loader knew the shard, its size and
the host's headroom at the moment it failed, and printed none of them. These tests hold
that surface still: inert by default, exact when armed, and — the part that actually
regresses quietly — the exception the caller sees is the SAME exception, annotated.
"""

import os

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.loader import (  # noqa: E402
    LAUNCH_BLOCKING_ENV,
    SYNC_DEBUG_ENV,
    _explain_shard_failure,
    _gib,
    _host_mem_facts,
    _largest_shard,
    _LoadStageTrace,
    _shard_read_diagnosis,
)


@pytest.fixture
def no_sync(monkeypatch):
    """Count synchronise calls without a GPU: the trace's contract is WHEN it syncs."""
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append(a))
    return calls


# --------------------------------------------------------------------------- inert


def test_trace_is_inert_without_the_flag(monkeypatch, no_sync):
    """Default OFF. The synchronisations serialise a load that is fine on most hosts —
    a debug mode that costs everyone time to help two hosts is not worth shipping."""
    monkeypatch.delenv(SYNC_DEBUG_ENV, raising=False)
    trace = _LoadStageTrace("cuda")
    assert trace.enabled is False
    trace.arm()
    trace.phase("quantise experts: layer 0")
    trace.enter("read 'x'")
    trace.tick()
    trace.sync("map shard handles")
    assert no_sync == [], "an inert trace must not touch the GPU at all"
    assert trace.last is None and trace.current is None


def test_inert_where_points_at_the_flag(monkeypatch):
    """The failure path still has to be useful when the flag was off — that is the run
    that actually happens first, and it is the one that must say what to do next."""
    monkeypatch.delenv(SYNC_DEBUG_ENV, raising=False)
    where = _LoadStageTrace("cuda").where()
    assert SYNC_DEBUG_ENV in where and LAUNCH_BLOCKING_ENV in where


# ------------------------------------------------------------------- arming


def test_arm_sets_launch_blocking_while_cuda_is_uninitialised(monkeypatch, no_sync):
    """CUDA_LAUNCH_BLOCKING is read when the context is created, so the loader can only
    set it before the first GPU touch. It arms at exactly that point, and this is the
    whole reason the flag is useful from inside the loader at all."""
    monkeypatch.setenv(SYNC_DEBUG_ENV, "1")
    monkeypatch.delenv(LAUNCH_BLOCKING_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    _LoadStageTrace("cuda").arm()
    assert os.environ[LAUNCH_BLOCKING_ENV] == "1"


def test_arm_does_not_pretend_after_cuda_is_up(monkeypatch, no_sync, caplog):
    """The dangerous outcome is a debug mode that looks armed and is not: setting the
    variable after context creation changes nothing, and must not be reported as if it
    had. The banner says so instead."""
    monkeypatch.setenv(SYNC_DEBUG_ENV, "1")
    monkeypatch.delenv(LAUNCH_BLOCKING_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    lines = []
    monkeypatch.setattr("experts4bit_qlora.loader.log", lines.append)
    _LoadStageTrace("cuda").arm()
    assert LAUNCH_BLOCKING_ENV not in os.environ, "must not set a variable that cannot bite"
    banner = "\n".join(lines)
    assert "NOT set" in banner and "cannot be turned on now" in banner


def test_a_cpu_device_disarms_rather_than_crashes(monkeypatch, no_sync):
    """torch.cuda.synchronize raises on a non-CUDA device. This flag is turned on by someone
    whose load is ALREADY failing; a version of it that can itself be the failure would be
    worse than no flag at all. It disarms and says so."""
    monkeypatch.setenv(SYNC_DEBUG_ENV, "1")
    lines = []
    monkeypatch.setattr("experts4bit_qlora.loader.log", lines.append)
    trace = _LoadStageTrace("cpu")
    assert trace.enabled is False
    assert any("is not a CUDA device" in ln for ln in lines)
    trace.phase("x")
    trace.enter("y")
    trace.tick()
    trace.sync("z")
    assert no_sync == []


def test_arm_reports_a_launch_blocking_the_caller_already_set(monkeypatch, no_sync):
    monkeypatch.setenv(SYNC_DEBUG_ENV, "1")
    monkeypatch.setenv(LAUNCH_BLOCKING_ENV, "1")
    lines = []
    monkeypatch.setattr("experts4bit_qlora.loader.log", lines.append)
    _LoadStageTrace("cuda").arm()
    assert any("already set in the environment" in ln for ln in lines)


# ------------------------------------------------------------- stage bookkeeping


def test_sync_and_tick_bound_the_failure(monkeypatch, no_sync):
    monkeypatch.setenv(SYNC_DEBUG_ENV, "1")
    monkeypatch.setattr("experts4bit_qlora.loader.log", lambda *a: None)
    trace = _LoadStageTrace("cuda")
    trace.enter("map shard handles")
    trace.sync("map shard handles")
    assert trace.last == "map shard handles"
    trace.phase("quantise experts: layer 3")
    trace.enter("read 'gate_up_proj' from shard-1")
    trace.tick()
    assert trace.last == "read 'gate_up_proj' from shard-1"
    assert len(no_sync) == 2


def test_a_fault_between_reads_names_its_phase(monkeypatch, no_sync):
    """The regression this guards is subtle and is exactly #344's own hypothesis: a
    kernel that runs BETWEEN two tensor reads (the expert quantisation) faults, and is
    observed at the next read. If `tick()` cleared the stage instead of falling back to
    the phase, that fault would report '<between stages>' and name nothing — the same
    silence the issue already spent three weeks in."""
    monkeypatch.setenv(SYNC_DEBUG_ENV, "1")
    monkeypatch.setattr("experts4bit_qlora.loader.log", lambda *a: None)
    trace = _LoadStageTrace("cuda")
    trace.phase("quantise experts: layer 17")
    trace.enter("read 'down_proj' from shard-0")
    trace.tick()                       # the read completed clean; the quantise follows
    where = trace.where()
    assert "quantise experts: layer 17" in where
    assert "<between stages>" not in where


# --------------------------------------------------------------- host facts


def test_host_mem_facts_never_raises_and_has_three_slots():
    """Reads /proc and /sys, which do not exist on every platform this package imports
    on. It describes a host; it must not be able to break a load."""
    total, avail, limit = _host_mem_facts()
    for value in (total, avail, limit):
        assert value is None or (isinstance(value, int) and value > 0)


def test_gib_says_unknown_rather_than_zero():
    """A missing reading printed as '0.0 GiB' is a false fact about the host, and this
    diagnosis exists to be read by someone deciding whether the host is the cause."""
    assert _gib(None) == "unknown"
    assert _gib(2 * 2**30) == "2.0 GiB"


def test_largest_shard_picks_the_biggest(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "b.safetensors").write_bytes(b"x" * 4096)
    name, size = _largest_shard(str(tmp_path), {"t1": "a.safetensors", "t2": "b.safetensors"})
    assert (name, size) == ("b.safetensors", 4096)


def test_largest_shard_tolerates_a_missing_file(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"x" * 10)
    name, _ = _largest_shard(str(tmp_path), {"t1": "a.safetensors", "t2": "gone.safetensors"})
    assert name == "a.safetensors"


# --------------------------------------------------------------- the diagnosis


def test_diagnosis_names_shard_size_host_and_the_issue(tmp_path):
    (tmp_path / "big.safetensors").write_bytes(b"x" * 8192)
    lines = _shard_read_diagnosis(
        "model.layers.0.mlp.experts.gate_up_proj", "big.safetensors",
        str(tmp_path), {"model.layers.0.mlp.experts.gate_up_proj": "big.safetensors"}, "cuda")
    text = "\n".join(lines)
    assert "gate_up_proj" in text and "big.safetensors" in text
    assert "MemTotal" in text and "cgroup limit" in text
    assert "issues/344" in text


def test_diagnosis_covers_the_mapping_failure_too(tmp_path):
    """The 30 GiB host died in safe_open itself, before any tensor existed to name. Same
    facts, different sentence — the two ends of one failure family."""
    (tmp_path / "big.safetensors").write_bytes(b"x" * 8192)
    lines = _shard_read_diagnosis(None, "big.safetensors", str(tmp_path),
                                  {"t": "big.safetensors"}, "cuda")
    assert any("MAPPING" in ln for ln in lines)


def test_explain_annotates_and_never_replaces_the_exception(tmp_path, monkeypatch):
    """A caller catching RuntimeError must keep catching RuntimeError. The loader's job
    here is to add facts, not to re-type someone else's error."""
    (tmp_path / "s.safetensors").write_bytes(b"x" * 64)
    logged = []
    monkeypatch.setattr("experts4bit_qlora.loader.log", logged.append)
    monkeypatch.delenv(SYNC_DEBUG_ENV, raising=False)
    exc = RuntimeError("CUDA error: invalid argument")
    _explain_shard_failure(exc, "w", "s.safetensors", str(tmp_path),
                           {"w": "s.safetensors"}, "cuda", _LoadStageTrace("cuda"))
    assert type(exc) is RuntimeError
    assert str(exc) == "CUDA error: invalid argument", "the message itself is untouched"
    assert any("issues/344" in ln for ln in logged), "the log is what receipts capture"
    if hasattr(exc, "add_note"):        # 3.11+
        assert any("issues/344" in n for n in exc.__notes__)
