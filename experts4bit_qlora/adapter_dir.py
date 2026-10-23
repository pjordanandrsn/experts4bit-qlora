# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""A trained adapter directory, verified and merged for serving (``backend-adapter/1``; docs/ADAPTER_PORTABILITY.md).

Loggetta's ``--adapter-out`` writes ``adapter.safetensors`` and ``adapter_manifest.json`` (manifest last). The paged
server serves one such adapter by MERGING its attention LoRA into the bf16 projection weights at load, before the
levers, fusions and graph capture, so everything downstream sees plain ``nn.Linear``. Merging adds no per-token work
and no new module.

* :func:`verify_adapter_dir` checks the directory (paths, sizes, sha256, the tensor set, shapes, dtypes and
  finiteness) and reads the tensors with safetensors, never pickle. Any failure is refusal 7.
* :func:`servable_refusals` lists every reason this server cannot serve the adapter as trained. Each reason is
  numbered as in the spec, and the caller refuses at startup on the first one, never silently serving the base.
* :func:`merge_attention_lora` merges ``W <- W + (alpha/r) * B @ A`` per projection, in fp32 on the weight's device,
  then casts. It reports, per projection, the max |merged - (W + s*B*A)|, the one rounding the merge adds.

Exactness is per module. An attention adapter was trained inside a whole model whose experts were the training
recipe's. A v1 manifest does not record the expert absmax form, so the whole model is ``unverified`` there
(ADAPTER_PORTABILITY.md, F1).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_V1 = "backend-adapter/1"
SCHEMA_V2 = "backend-adapter/2"
_SHA256 = re.compile(r"[0-9a-f]{64}")
WEIGHTS = "adapter.safetensors"
MANIFEST = "adapter_manifest.json"
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
#: ``created_at`` as a backend-adapter/2 writer records it: ``YYYY-MM-DDTHH:MM:SSZ`` (UTC), matched in full
_UTC_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
#: attention projection LoRA tensors, as the training model names them (``lora.LoRALinear`` wraps the projection)
_ATTN = re.compile(r"(?P<module>.+\.(?:q_proj|k_proj|v_proj|o_proj|out_proj))\.lora_(?P<ab>[AB])")
#: expert LoRA tensors (``lora.ExpertsLoRA``)
_EXPERT = re.compile(r".+\.(?:gate_up|down)_lora_[AB]")


class AdapterRefused(ValueError):
    """Startup refusal ``code`` (1-8, numbered as in the spec), with the reason."""

    def __init__(self, code: int, reason: str):
        super().__init__(f"adapter refused ({code}): {reason}")
        self.code, self.reason = code, reason


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class AdapterDir:
    path: str
    manifest: dict
    #: name -> CPU tensor, as trained
    tensors: dict
    setup: dict
    #: ``resolved_model_revision``: the base commit the adapter was trained on (None when the manifest has none)
    revision: str | None
    #: sha256 of ``adapter.safetensors``: the adapter's identity
    adapter_id: str
    manifest_sha256: str
    #: ``alpha / r`` from the manifest's setup, the scale training applied
    scale: float
    #: module path -> {"A": name, "B": name} for each attention projection
    attention: dict = field(default_factory=dict)
    expert_tensors: tuple = ()
    other_tensors: tuple = ()


def verify_adapter_dir(path) -> AdapterDir:
    """The directory as Loggetta wrote it, or :class:`AdapterRefused` (7) naming what failed."""
    import torch
    from safetensors.torch import load_file

    def bad(why):
        return AdapterRefused(7, f"{path}: {why}")

    root = Path(path).expanduser()
    if not root.is_dir() or root.is_symlink():
        raise bad("not a directory (or a symlink)")
    mpath = root / MANIFEST
    if not mpath.is_file() or mpath.is_symlink():
        raise bad(f"no {MANIFEST}")
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise bad(f"{MANIFEST} is not JSON ({e})") from None
    schema = manifest.get("schema") if isinstance(manifest, dict) else None
    if schema == SCHEMA_V2 and manifest.get("backend") != "experts4bit":
        raise bad(f"{SCHEMA_V2} is defined for backend=experts4bit only; a {manifest.get('backend')!r} /2 manifest is "
                  f"refused until a section for that backend is added (docs/ADAPTER_PORTABILITY.md section 6)")
    if schema not in (SCHEMA_V1, SCHEMA_V2) or manifest.get("backend") != "experts4bit":
        raise bad(f"schema {schema!r} / backend: this server reads {SCHEMA_V1} and {SCHEMA_V2} from the experts4bit "
                  f"backend")
    files = manifest.get("files")
    if manifest.get("weights") != WEIGHTS or not isinstance(files, dict) or WEIGHTS not in files:
        raise bad(f"the manifest must name {WEIGHTS} and list it in files")
    real = root.resolve()
    for name, info in files.items():
        f = root / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or f.is_symlink()
                or not f.resolve().is_relative_to(real)):
            raise bad(f"unsafe file path {name!r}")
        if not isinstance(info, dict) or not f.is_file() or f.stat().st_size != info.get("bytes") \
                or _sha256(f) != info.get("sha256"):
            raise bad(f"{name}: size or sha256 does not match the manifest")
    try:
        tensors = load_file(str(root / WEIGHTS), device="cpu")
    except Exception as e:                                    # noqa: BLE001 - any unreadable file is one refusal
        raise bad(f"{WEIGHTS} does not read as safetensors ({type(e).__name__})") from None
    recorded = manifest.get("tensors")
    if not isinstance(recorded, dict) or set(recorded) != set(tensors):
        raise bad("the safetensors tensor names differ from the manifest's")
    for name, t in tensors.items():
        r = recorded[name]
        if r.get("shape") != list(t.shape) or r.get("dtype") != str(t.dtype):
            raise bad(f"{name}: shape/dtype {list(t.shape)}/{t.dtype} vs the manifest's {r.get('shape')}/{r.get('dtype')}")
        if not torch.isfinite(t).all():
            raise bad(f"{name}: non-finite values")
    setup = manifest.get("setup")
    if not isinstance(setup, dict):
        raise bad("the manifest has no setup")
    r_, alpha = setup.get("r"), setup.get("alpha")
    if not (isinstance(r_, int) and not isinstance(r_, bool) and r_ > 0) or not isinstance(alpha, (int, float)) \
            or isinstance(alpha, bool):
        raise bad(f"setup r={r_!r} alpha={alpha!r}: the scale alpha/r cannot be read")
    attention, experts, other = {}, [], []
    for name in sorted(tensors):
        m = _ATTN.fullmatch(name)
        if m:
            attention.setdefault(m["module"], {})[m["ab"]] = name
        elif _EXPERT.fullmatch(name):
            experts.append(name)
        else:
            other.append(name)
    for module, ab in attention.items():
        if set(ab) != {"A", "B"}:
            raise bad(f"{module}: lora_A and lora_B must come together")
        a, b = tensors[ab["A"]], tensors[ab["B"]]
        if a.dim() != 2 or b.dim() != 2 or a.shape[0] != r_ or b.shape[1] != r_:
            raise bad(f"{module}: lora_A {list(a.shape)} / lora_B {list(b.shape)} are not rank {r_}")
    if schema == SCHEMA_V2:
        _verify_v2_fields(manifest, attention, float(alpha) / r_, bad)
    revision = manifest.get("resolved_model_revision")
    return AdapterDir(path=str(root), manifest=manifest, tensors=tensors, setup=setup,
                      revision=revision if isinstance(revision, str) and revision else None,
                      adapter_id=files[WEIGHTS]["sha256"], manifest_sha256=_sha256(mpath),
                      scale=float(alpha) / r_, attention=attention, expert_tensors=tuple(experts),
                      other_tensors=tuple(other))


_HF_SNAPSHOT = re.compile(r"(?:.*[/\\])?models--(?P<org>[^/\\]+?)--(?P<name>[^/\\]+)[/\\]snapshots[/\\]"
                          r"(?P<rev>[0-9a-f]{40})[/\\]?")


def arena_source(arena_index: dict) -> tuple:
    """``(repo_id, commit)`` an NF4 arena was baked from, when its index can show it: the HF cache snapshot directory it
    names (``.../models--<org>--<name>/snapshots/<sha>``). ``(None, None)`` when it cannot (a local path, or no
    snapshot recorded)."""
    snap = arena_index.get("snapshot") if isinstance(arena_index, dict) else None
    m = _HF_SNAPSHOT.fullmatch(str(snap)) if snap else None
    return (f"{m['org']}/{m['name']}", m["rev"]) if m else (None, None)


def _verify_v2_fields(manifest, attention, scale, bad) -> None:
    """``backend-adapter/2``'s added fields are present and well formed: every field adapter_schema.V2_FIELDS names, the
    producing run named (a non-empty ``run_id``) and dated (a real UTC ``created_at``), a known fingerprint scheme per
    scope, and the recorded applied scale equal to the setup's alpha/r."""
    from datetime import datetime

    from .adapter_schema import KNOWN_SCHEMES, SCOPES, V2_FIELDS

    missing = sorted(k for k in V2_FIELDS if k not in manifest)
    if missing:
        raise bad(f"{SCHEMA_V2} manifest lacks {missing}")
    # present is not enough: a null run_id or an unreadable time would pass a presence check and say nothing
    run_id = manifest["run_id"]
    if not (isinstance(run_id, str) and run_id.strip()):
        raise bad(f"run_id {run_id!r} does not name the run that produced the adapter")
    stamp = manifest["created_at"]
    try:
        ok = isinstance(stamp, str) and bool(_UTC_STAMP.fullmatch(stamp)) and bool(
            datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:                                     # the right shape but not a date (a 13th month, a 30 February)
        ok = False
    if not ok:
        raise bad(f"created_at {stamp!r} is not a UTC time as YYYY-MM-DDTHH:MM:SSZ")
    fps = manifest["fingerprints"]
    if not isinstance(fps, dict) or not set(fps) <= set(SCOPES):
        raise bad(f"fingerprints must map scopes from {list(SCOPES)}, got {sorted(fps) if isinstance(fps, dict) else fps!r}")
    # the scopes the adapter's own content requires: experts always (an experts4bit model has expert stacks), and
    # attention whenever the adapter carries attention LoRA
    required = ["experts"] + (["attention"] if attention else [])
    absent = [s for s in required if s not in fps]
    if absent:
        raise bad(f"fingerprints lacks the {absent} scope(s) this adapter requires")
    for scope, fp in fps.items():
        if not isinstance(fp, dict) or fp.get("scheme_id") not in KNOWN_SCHEMES:
            raise bad(f"fingerprints.{scope}: scheme {fp.get('scheme_id') if isinstance(fp, dict) else None!r} is not "
                      f"one this server knows ({list(KNOWN_SCHEMES)}); it refuses rather than compare unlike hashes")
        if not (isinstance(fp.get("sha256"), str) and _SHA256.fullmatch(fp["sha256"])):
            raise bad(f"fingerprints.{scope}.sha256 is not a sha256")
        mods = fp.get("modules")
        if not (isinstance(mods, int) and not isinstance(mods, bool) and mods > 0):
            raise bad(f"fingerprints.{scope}.modules {mods!r} is not a positive count")
        secs = fp.get("seconds")
        if not (isinstance(secs, (int, float)) and not isinstance(secs, bool) and secs >= 0):
            raise bad(f"fingerprints.{scope}.seconds {secs!r} is not a measured duration")
        if "fp32_source" in fp:
            _verify_fp32_source(fp, scope, bad)
    applied = manifest["lora_scale_applied"]
    held = applied.get("attention") if isinstance(applied, dict) else None
    if attention and (held is None or abs(float(held) - scale) > 1e-12):
        raise bad(f"lora_scale_applied.attention {held!r} is not the setup's alpha/r {scale!r}")


def _verify_fp32_source(fp, scope, bad) -> None:
    """An expert fingerprint's optional ``fp32_source`` (the base's qfp-v1 with each compressed stack's fp32 absmax,
    recorded before compression) is well formed: a sha256, and one sha256 per line of the scope, keyed by stack."""
    src = fp["fp32_source"]
    if scope != "experts" or not isinstance(src, dict):
        raise bad(f"fingerprints.{scope}.fp32_source is defined for the experts scope only, as an object")
    if not (isinstance(src.get("sha256"), str) and _SHA256.fullmatch(src["sha256"])):
        raise bad(f"fingerprints.{scope}.fp32_source.sha256 is not a sha256")
    by = src.get("absmax_by_stack")
    if not (isinstance(by, dict) and len(by) == fp["modules"]
            and all(isinstance(k, str) and isinstance(v, str) and _SHA256.fullmatch(v) for k, v in by.items())):
        raise bad(f"fingerprints.{scope}.fp32_source.absmax_by_stack must map each of the scope's {fp['modules']} "
                  f"lines to a sha256")


def arena_source_revision(arena_index: dict) -> str | None:
    return arena_source(arena_index)[1]


#: the base scopes an operator may accept unverified (``E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE``)
UNVERIFIED_BASE_SCOPES = ("experts",)
#: the base scopes an operator may serve despite a MEASURED fingerprint mismatch
#: (``E4B_PAGED_ADAPTER_ACCEPT_BASE_MISMATCH``); attention has none: its mismatch (11) is never overridden
BASE_MISMATCH_SCOPES = ("experts",)
ACCEPT_BASE_MISMATCH_ENV = "E4B_PAGED_ADAPTER_ACCEPT_BASE_MISMATCH"
#: the expert status when the manifest's fp32 source fingerprint, not the trained-form one, equals the arena's
SOURCE_VERIFIED = "verified (source)"


def parse_accept_unverified_base(raw: str) -> frozenset:
    """``E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE``: a comma-separated list of scopes, each named. Empty means none. A
    bare ``1``/``true`` or an unknown scope is an error: the operator names what they accept."""
    scopes = frozenset(s.strip().lower() for s in (raw or "").split(",") if s.strip())
    unknown = sorted(scopes - set(UNVERIFIED_BASE_SCOPES))
    if unknown:
        raise ValueError(f"E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE={raw!r}: unknown scope(s) {unknown}; name the scope "
                         f"to accept, one of {list(UNVERIFIED_BASE_SCOPES)}")
    return scopes


def parse_accept_base_mismatch(raw: str) -> frozenset:
    """``E4B_PAGED_ADAPTER_ACCEPT_BASE_MISMATCH``: a comma-separated list of scopes, each named (only ``experts``). It
    serves an adapter whose recorded expert fingerprint the served arena MEASURABLY does not match; it never covers an
    attention mismatch (refusal 11), and ``E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE`` never covers a measured one."""
    scopes = frozenset(s.strip().lower() for s in (raw or "").split(",") if s.strip())
    unknown = sorted(scopes - set(BASE_MISMATCH_SCOPES))
    if unknown:
        raise ValueError(f"{ACCEPT_BASE_MISMATCH_ENV}={raw!r}: unknown scope(s) {unknown}; name the scope to accept, "
                         f"one of {list(BASE_MISMATCH_SCOPES)} (an attention mismatch is never overridden)")
    return scopes


def _v2_scope_fingerprint(ad: AdapterDir, scope: str):
    """The manifest's recorded fingerprint for ``scope`` when it is one this server can compare (a /2 manifest, a known
    scheme), else None."""
    from .adapter_schema import KNOWN_SCHEMES

    if ad.manifest.get("schema") != SCHEMA_V2:
        return None
    fp = (ad.manifest.get("fingerprints") or {}).get(scope)
    return fp if isinstance(fp, dict) and fp.get("scheme_id") in KNOWN_SCHEMES else None


def _dq_trained(ad: AdapterDir) -> bool:
    return (ad.manifest.get("expert_absmax") or {}).get("form") == "double-quantized"


def fingerprint_check(ad: AdapterDir, model, arena: str, *, accept_base_mismatch: frozenset = frozenset(),
                      fingerprint=None) -> tuple:
    """``(refusals, report)``: the SERVED base's qfp-v1 against the adapter's recorded one, per scope the /2 manifest
    records. Called after the load and BEFORE :func:`merge_attention_lora` (the merge rewrites the attention weights)
    and before any lever. ``report[scope]`` is ``{"scheme", "trained", "served", "modules", "seconds", "status"}``,
    status ``verified`` | ``verified (source)`` | ``mismatch`` |
    ``mismatch (served under E4B_PAGED_ADAPTER_ACCEPT_BASE_MISMATCH=experts)``. ``verified (source)``: the expert
    fingerprint differs only because training compressed the absmax, and the manifest's ``fp32_source`` (the base's
    fingerprint with the fp32 absmax, recorded before compression) equals the arena's. It proves the served BASE is the
    trained one and serves without an override; the served absmax is fp32 and the trained one was double-quantized, a
    declared recipe difference whose quality effect is not read here. Refusal 11: the attention base differs (never
    overridden). Refusal 12: the expert base differs, unless the operator names ``experts`` in
    ``E4B_PAGED_ADAPTER_ACCEPT_BASE_MISMATCH``. ``fingerprint`` is
    :func:`.adapter_schema.quant_fingerprint` (a test may pass its own)."""
    if fingerprint is None:
        from .adapter_schema import quant_fingerprint as fingerprint
    refusals, report = [], {}
    for scope in ("attention", "experts"):
        want = _v2_scope_fingerprint(ad, scope)
        if want is None:
            continue
        got = (fingerprint(model, "attention", source="resident") if scope == "attention"
               else fingerprint(model, "experts", source="arena", arena=arena))
        match = got["sha256"] == want.get("sha256") and got.get("scheme_id") == want.get("scheme_id")
        row = {"scheme": got.get("scheme_id"), "trained": want.get("sha256"), "served": got["sha256"],
               "modules": {"trained": want.get("modules"), "served": got.get("modules")},
               "seconds": got.get("seconds"), "status": "verified" if match else "mismatch"}
        report[scope] = row
        if match:
            continue
        if scope == "attention":
            refusals.append((11, f"the served attention base is not the adapter's: qfp-v1 {got['sha256'][:16]} served "
                                 f"against {str(want.get('sha256'))[:16]} trained ({got.get('modules')} vs "
                                 f"{want.get('modules')} modules). The attention LoRA would be merged into weights it "
                                 f"was not trained on; this refusal has no override"))
            continue
        src = want.get("fp32_source") if isinstance(want.get("fp32_source"), dict) else None
        if src is not None and src.get("sha256") == got["sha256"] and got.get("scheme_id") == want.get("scheme_id"):
            # the base is the trained one: training compressed its absmax, and recorded the fp32 absmax first
            row["status"] = SOURCE_VERIFIED
            row["fp32_source"] = src["sha256"]
            row["trained_absmax"] = (ad.manifest.get("expert_absmax") or {}).get("form")
            continue
        if "experts" in accept_base_mismatch:
            row["status"] = f"mismatch (served under {ACCEPT_BASE_MISMATCH_ENV}=experts)"
        else:
            if src is not None:
                first = _first_stack_differing(src.get("absmax_by_stack"), got.get("absmax_by_stack"))
                why = (f" The adapter records its base's fp32 source fingerprint too ({str(src.get('sha256'))[:16]}) "
                       f"and the arena matches neither: it was not baked from the base this adapter was trained on, or not "
                       f"with the same recipe"
                       + (f" (the first stack whose absmax differs: {first})." if first else "."))
            elif _dq_trained(ad):
                why = (" The manifest records a double-quantized expert absmax (training's default) and no fp32 source "
                       "fingerprint, and the arena stores fp32: this is the dq-vs-fp32 absmax case "
                       "(docs/ADAPTER_PORTABILITY.md F1), not necessarily a different model. Retraining with this "
                       "release (it records the fp32 source fingerprint, so the base verifies) or at E4B_ABSMAX_DQ=0 "
                       "resolves it.")
            else:
                why = " The arena was not baked from the base this adapter was trained on (or with another recipe)."
            refusals.append((12, f"the served expert base is not the adapter's: qfp-v1 {got['sha256'][:16]} served "
                                 f"against {str(want.get('sha256'))[:16]} trained.{why} To serve it anyway, set "
                                 f"{ACCEPT_BASE_MISMATCH_ENV}=experts (reported in /health as a served mismatch); "
                                 f"E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE does not cover a measured mismatch"))
    return refusals, report


def _first_stack_differing(trained, served) -> str | None:
    """The first ``<stack>.<which>`` (sorted) whose recorded absmax hash differs between two ``absmax_by_stack``
    maps, or that one of them lacks; None when either map is missing."""
    if not (isinstance(trained, dict) and isinstance(served, dict)):
        return None
    return next((k for k in sorted(set(trained) | set(served)) if trained.get(k) != served.get(k)), None)


def _expert_base_unverified(ad: AdapterDir) -> str:
    """Why this server cannot verify ``ad``'s expert base, for its manifest's schema."""
    if ad.manifest.get("schema") == SCHEMA_V2:
        form = (ad.manifest.get("expert_absmax") or {}).get("form")
        return (f"backend-adapter/2 records the expert absmax form ({form}) but no expert fingerprint in a scheme this "
                f"server knows, so there is nothing to compare against its arena; the NF4 arena stores fp32 absmax "
                f"(docs/ADAPTER_PORTABILITY.md F1)")
    return ("backend-adapter/1 records no expert absmax form; training defaults to the double-quantized absmax, the NF4 "
            "arena stores fp32 (docs/ADAPTER_PORTABILITY.md F1)")


def base_match(ad: AdapterDir, refusals=(), fingerprints=None) -> dict:
    """What is known about the served base against the adapter's training base, per part of the model. ``fingerprints``
    is :func:`fingerprint_check`'s report, once the load has measured it."""
    fps = fingerprints or {}
    attn = ("refused" if any(code in (8, 11) for code, _ in refusals) else
            "exact by construction (up to the measured merge rounding)" if ad.attention else "none")
    if fps.get("attention", {}).get("status") == "verified":
        attn += f"; base verified (qfp-v1 {fps['attention']['served'][:16]})"
    ex = fps.get("experts")
    if ex is not None:
        if ex["status"] == "verified":
            experts = f"verified (qfp-v1 {ex['served'][:16]})"
        elif ex["status"] == SOURCE_VERIFIED:
            # the BASE is proven, never "served == trained": the served absmax is fp32, the trained one compressed
            experts = (f"verified (source, qfp-v1 {ex['served'][:16]}); trained on "
                       f"{ex.get('trained_absmax') or 'double-quantized'} absmax, served on fp32 absmax")
        else:
            experts = f"{ex['status']}: qfp-v1 {ex['served'][:16]} served, {str(ex['trained'])[:16]} trained"
    elif _v2_scope_fingerprint(ad, "experts") is not None:
        experts = "pending (the manifest's qfp-v1 is compared with the served arena at load)"
    else:
        experts = f"unverified ({_expert_base_unverified(ad)})"
    return {"attention": attn, "experts": experts}


def servable_refusals(ad: AdapterDir, *, served_revision: str | None, requested_revision: str | None,
                      arena_index: dict | None, levers: dict, allow_unverified_arena: bool = False,
                      accept_unverified_base: frozenset = frozenset(), check_served_revision: bool = True,
                      served_model: str | None = None, spec: str = "off") -> list:
    """Every ``(code, reason)`` that stops this server from serving ``ad`` as trained (empty: servable). The codes are
    the spec's: 1 expert LoRA, 2 base revision, 3 arena source (repo and revision), 4 arena recipe, 5 re-quantizing levers,
    6 unknown target modules, 8 attention trained on a 4-bit base (v1), 9 an expert base this server cannot verify (v1),
    unless the operator accepts it by name, 10 speculative decoding (its draft head and gate were read on the base); 11
    and 12 (the measured attention and expert fingerprints) come from :func:`fingerprint_check` after the load. Refusal 7 (the verifier) has already passed. ``check_served_revision=False`` skips only
    the served commit's comparison, for the check made before the load (the loader resolves the commit)."""
    import torch

    out = []
    if "experts" not in accept_unverified_base and _v2_scope_fingerprint(ad, "experts") is None:
        # v1 records no expert fingerprint (nor a /2 without one in a known scheme): nothing verifies the expert base.
        # A /2 with one is compared with the served arena at load (fingerprint_check: refusal 12, or verified).
        out.append((9, f"the expert base cannot be verified: {_expert_base_unverified(ad)}, so the whole served model "
                       f"may differ from the trained one. Set E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE=experts to serve "
                       f"it anyway (reported as base_match.experts=unverified)"))
    live = [n for n in ad.expert_tensors if n.endswith("_lora_B") and bool(torch.any(ad.tensors[n] != 0))]
    if live:
        out.append((1, f"expert LoRA is not served on the paged path ({len(live)} expert lora_B tensors are nonzero, "
                       f"e.g. {live[0]}); serve it with loggetta.load_adapter, or train attention-only "
                       f"(train_experts=false)"))
    if ad.other_tensors:
        out.append((6, f"unknown target modules ({len(ad.other_tensors)} tensors, e.g. {ad.other_tensors[0]}): only "
                       f"attention q/k/v/o LoRA and expert LoRA are known"))
    if not ad.revision:
        out.append((2, "the manifest records no resolved_model_revision: there is no base to verify against"))
    elif not check_served_revision:
        pass
    elif not served_revision:
        out.append((2, f"the served base's commit is unknown (a local directory?); the adapter was trained on "
                       f"{ad.revision}. Serve the hub id pinned to that revision"))
    elif served_revision != ad.revision:
        out.append((2, f"base revision {served_revision} is not the adapter's {ad.revision}"))
    if requested_revision and ad.revision and _FULL_SHA.fullmatch(requested_revision) \
            and requested_revision != ad.revision:
        out.append((2, f"E4B_PAGED_REVISION {requested_revision} is not the adapter's {ad.revision}"))
    arena_repo, arena_rev = arena_source(arena_index or {})
    if arena_rev is None:
        if not allow_unverified_arena:
            out.append((3, "the arena's source is unknown (its index records no HF snapshot commit); set "
                           "E4B_PAGED_ADAPTER_ALLOW_UNVERIFIED_ARENA=1 to serve anyway (reported in /health)"))
    else:
        if ad.revision and arena_rev != ad.revision:
            out.append((3, f"the arena was baked from {arena_rev}, not the adapter's base {ad.revision}"))
        if served_model and "/" in served_model and not os.path.isabs(served_model) and arena_repo != served_model:
            out.append((3, f"the arena was baked from {arena_repo}, not the served model {served_model}"))
    q = (arena_index or {}).get("quantizer")
    if not isinstance(q, dict) or not all(k in q for k in ("quant_type", "blocksize")):
        out.append((4, "cannot verify the arena's quantization recipe: its index has no quantizer block with "
                       "quant_type and blocksize"))
    else:
        for key in ("quant_type", "blocksize"):
            if key in ad.setup and q[key] != ad.setup[key]:
                out.append((4, f"arena {key} {q[key]!r} is not the adapter's {ad.setup[key]!r}"))
    on = sorted(k for k, v in levers.items() if v)
    if on:
        out.append((5, f"levers {on} re-quantize an adapted module or depend on calibration read on the base model; "
                       f"refused under an adapter until a registered read gates them (docs/ADAPTER_PORTABILITY.md F4)"))
    if (spec or "off") != "off":
        out.append((10, f"E4B_PAGED_SPEC={spec}: the draft head and its quality gate were read on the base model, not an "
                        f"adapted one; refused under an adapter until a registered read covers it"))
    if ad.setup.get("attn_4bit") and ad.attention:
        out.append((8, "the attention adapter was trained on a 4-bit attention base (attn_4bit=true); merging it into "
                       "bf16 W serves it on a base it never saw. Retrain with attn_4bit=false, or use a "
                       "backend-adapter/2 adapter whose base fingerprint the server can verify"))
    return out


@dataclass
class MergeReport:
    merged: int
    scale: float
    #: module path -> {"max_err": max |merged - (W + s*B*A)|, "max_abs_delta": max |s*B*A|, "shape": [out, in]}
    projections: dict

    def to_dict(self) -> dict:
        return {"merged": self.merged, "scale": self.scale,
                "max_err": max((p["max_err"] for p in self.projections.values()), default=0.0),
                "max_abs_delta": max((p["max_abs_delta"] for p in self.projections.values()), default=0.0),
                "projections": self.projections}


def merge_attention_lora(model, ad: AdapterDir) -> MergeReport:
    """Merge each attention projection's LoRA into its weight in place: ``W <- cast(W + scale * B @ A)``, computed in
    fp32 on the weight's device. Each target must be a plain ``nn.Linear`` (before any lever or fusion); anything else
    refuses (7), and nothing is merged until every target is checked."""
    import torch
    from torch import nn

    targets = {}
    for module, ab in ad.attention.items():
        try:
            lin = model.get_submodule(module)
        except AttributeError:
            raise AdapterRefused(7, f"{module}: no such module in the served model") from None
        if type(lin) is not nn.Linear:
            raise AdapterRefused(7, f"{module}: {type(lin).__name__}, not a plain nn.Linear (merge before levers and "
                                    f"fusions)")
        a, b = ad.tensors[ab["A"]], ad.tensors[ab["B"]]
        if (b.shape[0], a.shape[1]) != tuple(lin.weight.shape):
            raise AdapterRefused(7, f"{module}: B@A is {[b.shape[0], a.shape[1]]}, the weight is "
                                    f"{list(lin.weight.shape)}")
        targets[module] = (lin, a, b)
    out = {}
    with torch.no_grad():
        for module, (lin, a, b) in targets.items():
            w = lin.weight
            delta = ad.scale * (b.to(w.device, torch.float32) @ a.to(w.device, torch.float32))
            exact = w.float() + delta
            merged = exact.to(w.dtype)
            out[module] = {"max_err": float((merged.float() - exact).abs().max()),
                           "max_abs_delta": float(delta.abs().max()), "shape": list(w.shape)}
            w.copy_(merged)
            del delta, exact, merged
    return MergeReport(merged=len(targets), scale=ad.scale, projections=out)
