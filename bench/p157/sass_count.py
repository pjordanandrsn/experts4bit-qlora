# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Static issued-instruction count per weight in a kernel's inner K loop, from its SASS (no GPU, no timing).

A standing instrument for kernel PRs: compile a Triton kernel exactly as the JIT would for a served launch, for a named
GPU target, disassemble it, find the inner K loop, and report its lane instructions per weight by class. Every kernel
PR that changes a decode or a load path can quote one number per target.

**The JIT's specialization is part of the instrument.** A served launch specializes every tensor argument whose
data pointer is 16-byte aligned, and every int divisible by 16, to ``tt.divisibility = 16``
(``triton/runtime/jit.py`` ``specialize_impl`` -> the backend's ``parse_attr("D")``). Ints equal to 1 become constexprs.
Those hints are what let loads vectorize. Compiled without them, the same kernel scalarizes its loads and adds
per-element address arithmetic, an inflated count. An early K25 count read 34.5 lane instructions per weight that way,
against 24.1 as served. :func:`jit_attrs` builds the hints from the served call's own values, and the test fails if
they go missing.

**The toolchain is part of the count.** The SASS is triton's and its bundled ptxas's, and a newer one can move a count
further than the change being measured. K25's served loop on sm_120 is 24.1 lane instructions per weight under
triton 3.4.0 (ptxas 12.8.93, the served stack's) and 36.2 under 3.8.0 (ptxas 12.9.86), where it also exceeds the
unspecialized compile's 32.6. So every record names its triton and ptxas, and :func:`compare` refuses a ratio across
two toolchains.

**The loop.** The inner K loop is the backward branch whose body holds the MMAs. Per loop iteration, a program covers
``weights_per_iteration`` weights (BLOCK_N x KC for a weight-streaming GEMM) with ``lanes`` threads, so::

    lane instructions per weight = (instructions in the loop body) x lanes / weights_per_iteration

The count assumes one issue per scheduler per cycle. It is a static floor, not a timing.

usage::

    python sass_count.py --kernel-dir ../../kernel --config k25_served --out out/      # K25 at e4b's served plan
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys

#: SASS opcode classes, first match wins.
CLASSES = (
    ("mma", r"^(HMMA|IMMA|QMMA|OMMA|UTCHMMA|HGMMA)"),
    ("load_global", r"^(LDG|LDGSTS|LDGDEPBAR)"),
    ("shared", r"^(LDS|STS|LDSM)"),
    ("select", r"^(SEL|FSEL)"),
    ("logic_shift", r"^(LOP3|SHF|PRMT|LEA|IADD3|IADD|IMAD|ISETP|PLOP3|P2R|R2P|SGXT|BMSK|FLO|POPC)"),
    ("move", r"^(MOV|UMOV|S2R|CS2R|S2UR)"),
    ("fp", r"^(FMUL|FFMA|FADD|F2F|F2FP|I2F|F2I|HADD2|HMUL2|HFMA2|FMNMX)"),
    ("sync_ctrl", r"^(BAR|DEPBAR|BRA|WARPSYNC|BSYNC|BSSY|NOP|EXIT|ERRBAR|MEMBAR|SYNCS|YIELD)"),
)
_LINE = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)\b([^;]*);")
#: A branch's target. The predicate can be an operand: newer ptxas closes K25's sm_120 loop with ``BRA.U UP0, 0xb70``.
_BRA = re.compile(r"BRA[A-Z.]*\s+(?:!?U?P[T0-9]+\s*,\s*)?(?:`\()?(0x[0-9a-f]+)")


def classify(opcode: str) -> str:
    for name, pat in CLASSES:
        if re.match(pat, opcode):
            return name
    return "other"


def parse_sass(raw: str) -> list:
    """``[(address, opcode, text)]`` from raw ``cuobjdump -sass`` output (``/*0a10*/  [@P0] OPCODE ... ;``)."""
    out = []
    for ln in raw.splitlines():
        m = _LINE.match(ln)
        if m:
            out.append((int(m.group(1), 16), m.group(3), ln.strip()))
    return out


def backward_branches(rows: list) -> list:
    """``(target, branch)`` addresses of every branch whose target lies above it: the loops."""
    found = []
    for addr, op, text in rows:
        if op.startswith("BRA"):
            m = _BRA.search(text)
            if m and int(m.group(1), 16) < addr:
                found.append((int(m.group(1), 16), addr))
    return found


#: What keys the inner loop: "mma", a body holding an MMA (K25's K loop); "load", the weight stream, for a variant
#: without an MMA, such as an ablation that removes it. On served K25 both keys pick the same loop.
LOOP_KEYS = {"mma": "mma", "load": "load_global"}
#: Bytes a lane moves per global-load instruction, by the opcode's width suffix (no suffix: 32 bits).
_LOAD_WIDTH = {"128": 16, "64": 8, "U16": 2, "S16": 2, "U8": 1, "S8": 1}


def load_bytes(op: str) -> int:
    """Bytes one lane moves in one global-load instruction ``op`` (0 for the LDGDEPBAR barrier)."""
    if op.startswith("LDGDEPBAR"):
        return 0
    return next((_LOAD_WIDTH[p] for p in op.split(".")[1:] if p in _LOAD_WIDTH), 4)


def inner_loop(rows: list, key: str = "mma") -> dict | None:
    """The inner K loop. ``None`` when no loop holds the ``key`` class.

    - ``key="mma"``: of the loops whose body holds an MMA, the smallest, so an outer persistent loop over work items,
      if any, is not taken for it.
    - ``key="load"``: the weight stream. Of the loops whose body holds a global load, the one whose OWN load bytes
      per iteration (its body's, less those of the loops nested inside it) are the most. It is not the smallest: a
      small loop that loads indices (a gather over ``order``, a tile-table walk) also holds a load. Own bytes, not
      total bytes, keep an outer loop from winning on the K loop's bytes alone, and keep a weight loop that wraps an
      inner index loop from losing to it. They do NOT settle a pipelined persistent kernel, whose stage-ahead prologue
      inside the outer loop can out-load the K loop; there the outer loop can win, and :func:`unroll_by_bytes`
      refuses it (it wraps a nested loop), so a guarded count is never silent. The loops passed over are in the
      pick's ``rejected``, and the next one, with its loads, is its ``runner_up`` (``None`` when there is none), so a
      reader sees what lost. The pick records ``nested_loops``, how many loops sit inside it."""
    if key not in LOOP_KEYS:
        raise ValueError(f"loop key {key!r}: expected one of {sorted(LOOP_KEYS)}")
    loops = []
    for a, b in backward_branches(rows):
        body = [r for r in rows if a <= r[0] <= b]
        ops = collections.Counter(classify(r[1]) for r in body)
        if ops.get(LOOP_KEYS[key]):
            opcodes = collections.Counter(r[1] for r in body)
            loops.append({"target": hex(a), "branch": hex(b), "instructions": len(body), "classes": dict(ops),
                          "opcodes": dict(opcodes), "span": (a, b),
                          "load_bytes": sum(load_bytes(op) * n for op, n in opcodes.items()
                                            if classify(op) == "load_global")})
    if not loops:
        return None
    if key == "mma":
        pick = min(loops, key=lambda x: x["instructions"])
    else:
        every = backward_branches(rows)
        for lp in loops:
            a, b = lp["span"]
            inside = [(c, d) for c, d in every if a <= c and d <= b and (c, d) != (a, b)]
            nested = {r[0] for r in rows if any(c <= r[0] <= d for c, d in inside)}
            lp["nested_loops"] = len(inside)
            lp["own_load_bytes"] = sum(load_bytes(r[1]) for r in rows if a <= r[0] <= b and r[0] not in nested
                                       and classify(r[1]) == "load_global")
        ranked = sorted(loops, key=lambda x: (-x["own_load_bytes"], x["instructions"]))
        pick = ranked[0]
        keys = ("target", "branch", "instructions", "load_bytes", "own_load_bytes")
        pick["rejected"] = [{k: lp[k] for k in keys} for lp in ranked[1:]]
        pick["runner_up"] = ({k: ranked[1][k] for k in keys} |
                             {"loads": {op: n for op, n in sorted(ranked[1]["opcodes"].items())
                                        if classify(op) == "load_global"}}) if len(ranked) > 1 else None
    for lp in loops:
        del lp["span"]
    return pick


def per_weight(loop: dict, lanes: int, weights_per_iteration: int) -> dict:
    scale = lanes / weights_per_iteration
    return {"lane_instr_per_weight": loop["instructions"] * scale,
            "by_class": {c: n * scale for c, n in sorted(loop["classes"].items())}}


#: MMA instruction shapes (m, n, k) by the opcode's shape suffix: HMMA.1688 (TF32, m16n8k8), HMMA.16816 (bf16 / fp16,
#: m16n8k16). An MMA the guard does not know is refused rather than guessed.
MMA_SHAPES = {"1688": (16, 8, 8), "16816": (16, 8, 16)}


def unroll_factor(loop: dict, *, block_m: int, block_n: int, kc: int, num_warps: int) -> int:
    """How many K iterations the picked loop body holds. One iteration of a ``block_m x block_n x kc`` tile on
    ``num_warps`` warps issues ``(block_m/m)(block_n/n)(kc/k) / num_warps`` MMAs per warp for the shape (m, n, k).
    ptxas can unroll the K loop, and then the body holds k times that, so dividing by one iteration's weights would
    overstate the count k-fold without an error. Returns k >= 1, or raises on an unknown MMA shape or a count that is
    not a whole multiple of one iteration's."""
    mmas = {op: n for op, n in loop["opcodes"].items() if classify(op) == "mma"}
    shapes = set()
    for op in mmas:
        m = re.match(r"^HMMA\.(\d+)", op)
        if not m or m.group(1) not in MMA_SHAPES:
            raise ValueError(f"unroll guard: unknown MMA {op!r}; known shapes {sorted(MMA_SHAPES)}")
        shapes.add(MMA_SHAPES[m.group(1)])
    if len(shapes) != 1:
        raise ValueError(f"unroll guard: expected one MMA shape in the loop, got {sorted(shapes)}")
    mm, nn, kk = shapes.pop()
    per_iter, rem = divmod((block_m // mm) * (block_n // nn) * (kc // kk), num_warps)
    total = sum(mmas.values())
    if rem or per_iter < 1 or total % per_iter:
        raise ValueError(f"unroll guard: {total} MMAs per warp in the loop is not a whole number of iterations of "
                         f"{(block_m // mm) * (block_n // nn) * (kc // kk)} / {num_warps} warps")
    return total // per_iter


def jit_attrs(fn, signature: dict, values: dict, backend) -> dict:
    """The served JIT launch's specialization, as ``ASTSource`` attrs: ``tt.divisibility = 16`` on every pointer
    argument (torch allocations are 16-byte aligned) and on every int in ``values`` divisible by 16, through the
    backend's own ``parse_attr("D")``. An int equal to 1 must be passed as a constexpr by the caller, as the JIT
    does."""
    for name, v in values.items():
        if isinstance(v, int) and v == 1:
            raise ValueError(f"{name} == 1: the JIT makes it a constexpr; pass it in constexprs")
    div = [n for n, t in signature.items() if t.startswith("*")]
    div += [n for n, v in values.items() if isinstance(v, int) and v % 16 == 0]
    return {(fn.arg_names.index(n),): backend.parse_attr("D") for n in div}


def compile_and_count(fn, signature: dict, constexprs: dict, values: dict, *, arch: int, num_warps: int,
                      num_stages: int, weights_per_iteration: int, out_dir: str | None = None, tag: str = "",
                      specialize: bool = True, tile: tuple | None = None, loop_key: str = "mma",
                      bytes_per_iteration: int | None = None) -> dict:
    """Compile ``fn`` for ``sm_<arch>`` as the JIT would (``specialize=False`` only to show what is lost), count its
    inner K loop, and return the record. Writes the SASS, PTX and cubin to ``out_dir`` when given.

    ``tile`` = (BLOCK_M, BLOCK_N, KC) turns on the unroll guard (:func:`unroll_factor`). A loop body that holds k K
    iterations counts against k x ``weights_per_iteration``, and k is recorded as ``unroll``. A count that is not a
    whole number of iterations is refused. ``loop_key`` picks how the inner loop is found (:func:`inner_loop`), and
    ``bytes_per_iteration`` turns on the guard for a loop with no MMA (:func:`unroll_by_bytes`)."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.compiler import make_backend
    from triton.tools.disasm import path_to_cuobjdump

    target = GPUTarget("cuda", arch, 32)
    attrs = jit_attrs(fn, signature, values, make_backend(target)) if specialize else {}
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    k = triton.compile(src, target=target, options={"num_warps": num_warps, "num_stages": num_stages})
    cubin = k.asm["cubin"]
    tmp = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tmp = os.path.join(out_dir, f"{tag or fn.__name__}.sm{arch}.cubin")
    else:
        import tempfile
        tmp = tempfile.mkstemp(suffix=".cubin")[1]
    with open(tmp, "wb") as f:
        f.write(cubin)
    raw = subprocess.run([path_to_cuobjdump(), "-sass", tmp], capture_output=True, text=True, check=True).stdout
    if out_dir:
        with open(tmp.replace(".cubin", ".sass"), "w", encoding="utf-8", newline="\n") as f:
            f.write(raw)
        with open(tmp.replace(".cubin", ".ptx"), "w", encoding="utf-8", newline="\n") as f:
            f.write(k.asm["ptx"])
    else:
        os.remove(tmp)
    rows = parse_sass(raw)
    rec = {"arch": f"sm_{arch}", **toolchain(), "specialized": specialize,
           "divisible_args": sorted(signature_name(fn, i) for (i,) in attrs), "total_instructions": len(rows)}
    rec.update(loop_record(rows, num_warps=num_warps, weights_per_iteration=weights_per_iteration, tile=tile,
                           loop_key=loop_key, bytes_per_iteration=bytes_per_iteration))
    return rec


def toolchain() -> dict:
    """The compiler behind a count: triton's version and its bundled ptxas's (``ptxas`` is None if it cannot be
    read). Both move counts, so every record carries them."""
    import triton
    ptxas = None
    try:
        from triton.backends.nvidia.compiler import get_ptxas_version
        m = re.search(r"V(\d+\.\d+\.\d+)", get_ptxas_version())
        ptxas = m.group(1) if m else None
    except Exception:                                   # an older or newer backend without the helper: unknown
        pass
    return {"triton": triton.__version__, "ptxas": ptxas}


def _one_toolchain(a: dict, b: dict) -> None:
    for key in ("triton", "ptxas"):
        if a.get(key) is None or a.get(key) != b.get(key):
            raise ValueError(f"refusing to compare counts across toolchains: {key} {a.get(key)!r} vs {b.get(key)!r}")


def compare(a: dict, b: dict) -> float:
    """``a``'s lane instructions per weight over ``b``'s, for two records of ONE toolchain (the same triton and ptxas,
    both known). Across toolchains the ratio is refused, not computed: a triton bump moved K25's served count on sm_120
    from 24.1 (3.4.0) to 36.2 (3.8.0), more than the decode changes this instrument is for."""
    _one_toolchain(a, b)
    return a["lane_instr_per_weight"] / b["lane_instr_per_weight"]


def traffic_signature(loop: dict, lanes: int, weights_per_iteration: int, cls: str) -> dict:
    """The loop's instructions of class ``cls`` per weight, by opcode, so each count carries its width."""
    scale = lanes / weights_per_iteration
    return {op: n * scale for op, n in sorted(loop["opcodes"].items()) if classify(op) == cls}


def load_signature(loop: dict, lanes: int, weights_per_iteration: int) -> dict:
    """The loop's global loads per weight (``LDG.E.128``, ``LDGSTS.E.BYPASS.128``, ``LDG.E.U8`` ...)."""
    return traffic_signature(loop, lanes, weights_per_iteration, "load_global")


def shared_signature(loop: dict, lanes: int, weights_per_iteration: int) -> dict:
    """The loop's shared-memory traffic per weight (``LDSM.16.M88.4``, ``LDS.64``, ``STS.128`` ...): the operand
    staging an ablation may claim to keep."""
    return traffic_signature(loop, lanes, weights_per_iteration, "shared")


def _same(a: dict, b: dict, key: str, what: str) -> bool:
    _one_toolchain(a, b)
    if a.get("arch") != b.get("arch"):
        raise ValueError(f"refusing to compare {what} across targets: {a.get('arch')!r} vs {b.get('arch')!r}")
    if a[key] == b[key]:
        return True
    # an unchecked side whose every count is the same whole multiple of the other's reads as an unroll, not a
    # different kernel: refuse rather than answer False
    for x, y in ((a, b), (b, a)):
        if x.get("unroll") is None and x[key] and set(x[key]) == set(y[key]) and all(y[key].values()):
            ratios = {round(x[key][op] / y[key][op], 9) for op in x[key]}
            if len(ratios) == 1 and (k := ratios.pop()) > 1 and float(k).is_integer():
                raise ValueError(f"{what}: an unchecked loop reads {int(k)}x the other's, as if unrolled "
                                 f"{int(k)}-fold: recount it with bytes_per_iteration")
    return False


def same_loads(a: dict, b: dict) -> bool:
    """Whether two records' inner loops issue the same global loads (count and width) per weight: an ablation of a
    kernel must keep every load of its control, or it is a different kernel. Refused across toolchains, as
    :func:`compare` is, and across targets."""
    return _same(a, b, "loads_per_weight", "loads")


def same_shared(a: dict, b: dict) -> bool:
    """Whether two records' inner loops issue the same shared-memory traffic (count and width) per weight: the check
    behind a claim that a variant keeps its control's operand staging. Refused as :func:`same_loads` is."""
    return _same(a, b, "shared_per_weight", "shared-memory traffic")


def opcode_signature(rec: dict, *, classes: tuple = (), prefixes: tuple = ()) -> dict:
    """A record's inner-loop opcodes per weight, those whose class is in ``classes`` or whose name starts with one of
    ``prefixes``. A parity check names its own set: "the same math" for a variant that drops K25's weight loads is the
    MMAs plus the instructions only the decode issues (for the select tree, ``FSEL`` and the absmax ``FMUL``). The
    ``logic_shift`` class mixes the decode's nibble work with load-address arithmetic, so a parity set should name
    opcodes rather than take that whole class."""
    loop = rec["inner_loop"]
    scale = rec["lanes"] / (rec["weights_per_iteration"] * (rec.get("unroll") or 1))
    return {op: n * scale for op, n in sorted(loop["opcodes"].items())
            if classify(op) in classes or op.startswith(tuple(prefixes))}


def same_signature(a: dict, b: dict, *, classes: tuple = (), prefixes: tuple = ()) -> bool:
    """Whether two records issue the same opcodes per weight in the named set (:func:`opcode_signature`), such as a
    compute-only variant's math against its control's. Refused across toolchains and targets; an empty set is refused,
    since it would always match."""
    if not classes and not prefixes:
        raise ValueError("an empty opcode set always matches: name classes or prefixes")
    _one_toolchain(a, b)
    if a.get("arch") != b.get("arch"):
        raise ValueError(f"refusing to compare opcodes across targets: {a.get('arch')!r} vs {b.get('arch')!r}")
    return (opcode_signature(a, classes=classes, prefixes=prefixes)
            == opcode_signature(b, classes=classes, prefixes=prefixes))


def function_loads(rows: list) -> dict:
    """Global-load opcodes in ``rows`` as raw counts, not per weight: over a whole function (``loads_in_function``) or
    outside its loop (``loads_outside_loop``). A variant whose loop ptxas unrolled away entirely has no loop to compare,
    so its loads are checked as totals: the function's equal the control's per-iteration loads x the trip count, plus
    the control's loads outside its loop."""
    return dict(sorted(collections.Counter(r[1] for r in rows if classify(r[1]) == "load_global").items()))


def unroll_by_bytes(loop: dict, *, lanes: int, bytes_per_iteration: int) -> int:
    """The unroll guard for a loop without MMAs: how many iterations of a stated ``bytes_per_iteration`` global-load
    stream its body moves (its load bytes per lane x ``lanes``). ptxas unrolls such loops when the trip count is known
    at compile time, as it is for any variant built to K25's signature (K is a constexpr). Returns k >= 1, or raises on
    a body that is no whole number of iterations, or that wraps a nested loop (its bytes per iteration are not
    static)."""
    if loop.get("nested_loops"):
        raise ValueError(f"unroll guard by bytes: the loop wraps {loop['nested_loops']} nested loop(s); "
                         "its bytes per iteration are not static")
    moved = loop["load_bytes"] * lanes
    k, rem = divmod(moved, bytes_per_iteration)
    if rem or k < 1:
        raise ValueError(f"unroll guard by bytes: {moved} bytes in the loop is not a whole number of "
                         f"{bytes_per_iteration}-byte iterations")
    return k


def stream_bytes(rec: dict) -> int:
    """The global-load bytes one iteration of a CHECKED record's loop moves (load bytes per lane x lanes / unroll):
    the ``bytes_per_iteration`` to count a variant against, taken from its control rather than from a formula (K25's
    served loop moves 3,584 bytes an iteration, more than its 3,200-byte stream, since some lanes load redundantly)."""
    if rec.get("unroll") is None:
        raise ValueError("stream_bytes: the record's unroll is unchecked")
    moved = rec["inner_loop"]["load_bytes"] * rec["lanes"]
    if moved % rec["unroll"]:
        raise ValueError(f"stream_bytes: {moved} bytes in the loop do not split into {rec['unroll']} iterations")
    return moved // rec["unroll"]


def loop_record(rows: list, *, num_warps: int, weights_per_iteration: int, tile: tuple | None = None,
                loop_key: str = "mma", bytes_per_iteration: int | None = None) -> dict:
    """The inner loop's record from parsed SASS, the loop found by ``loop_key`` (:data:`LOOP_KEYS`).

    The unroll guard runs one of two ways, and ``unroll_guard`` names which:
    - ``"mma"``: ``tile`` = (BLOCK_M, BLOCK_N, KC) given and the loop holds MMAs (:func:`unroll_factor`);
    - ``"bytes"``: the loop holds no MMA and ``bytes_per_iteration`` is given (:func:`unroll_by_bytes`; take it from
      the control with :func:`stream_bytes`).

    Otherwise ``unroll`` and ``unroll_guard`` are ``None``: the record never claims a check it did not run, and its
    per-weight figures assume one iteration per loop body, unchecked."""
    loop = inner_loop(rows, loop_key)
    lanes = num_warps * 32
    rec = {"lanes": lanes, "weights_per_iteration": weights_per_iteration, "loop_key": loop_key, "inner_loop": loop,
           "loads_in_function": function_loads(rows)}
    if loop:
        a, b = int(loop["target"], 16), int(loop["branch"], 16)
        rec["loads_outside_loop"] = function_loads([r for r in rows if not a <= r[0] <= b])
    if loop:
        unroll, guard = None, None
        if loop["classes"].get("mma"):
            if tile:
                unroll, guard = unroll_factor(loop, block_m=tile[0], block_n=tile[1], kc=tile[2],
                                              num_warps=num_warps), "mma"
        elif bytes_per_iteration:
            unroll, guard = unroll_by_bytes(loop, lanes=lanes, bytes_per_iteration=bytes_per_iteration), "bytes"
        rec["unroll"], rec["unroll_guard"] = unroll, guard
        rec.update(per_weight(loop, lanes, weights_per_iteration * (unroll or 1)))
        rec["loads_per_weight"] = load_signature(loop, lanes, weights_per_iteration * (unroll or 1))
        rec["shared_per_weight"] = shared_signature(loop, lanes, weights_per_iteration * (unroll or 1))
    return rec


def signature_name(fn, index: int) -> str:
    return fn.arg_names[index]


# --------------------------------------------------------------------------------------------- the built-in configs --

#: K25 (``nf4_smallm._gemm_nf4_grouped_smallm``) at experts4bit-qlora's served plan (``hot_residency._K25_PLAN``:
#: BLOCK_N 32, KC 64, 4 warps, 3 stages, the select-tree decode, fp32 weights through TF32 MMA) on Qwen3-30B-A3B's
#: W16 shapes, through the lean route (gather, ``gather_div`` = top-k on gate_up, scatter).
K25_SIGNATURE = {
    "x_ptr": "*bf16", "ord_ptr": "*i64", "w_ptr": "*u8", "am_ptr": "*fp32", "lut_ptr": "*fp32",
    "row0_ptr": "*i32", "rows_ptr": "*i32", "grp_ptr": "*i32", "out_ptr": "*bf16",
    "N": "i32", "stride_we": "i32", "stride_wn": "i32", "stride_ae": "i32", "stride_an": "i32",
    "K": "constexpr", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "KC": "constexpr", "LUT": "constexpr",
    "GATHER": "constexpr", "DOT_BF16": "constexpr", "EVEN_K": "constexpr", "sct_ptr": "*i64",
    "SCATTER": "constexpr", "GDIV": "constexpr"}
K25_PLAN = {"BLOCK_N": 32, "KC": 64, "num_warps": 4, "num_stages": 3, "LUT": 3, "DOT_BF16": False}
K25_SHAPES = {"gate_up": {"N": 1536, "K": 2048, "GDIV": 8}, "down": {"N": 2048, "K": 768, "GDIV": 1}}


def k25_served(shape: str) -> tuple:
    """``(signature, constexprs, values, weights_per_iteration)`` for K25's served call at ``shape``."""
    sh = K25_SHAPES[shape]
    n, k = sh["N"], sh["K"]
    consts = {"K": k, "BLOCK_M": 16, "BLOCK_N": K25_PLAN["BLOCK_N"], "KC": K25_PLAN["KC"], "LUT": K25_PLAN["LUT"],
              "GATHER": True, "DOT_BF16": K25_PLAN["DOT_BF16"], "EVEN_K": k % K25_PLAN["KC"] == 0, "SCATTER": True,
              "GDIV": sh["GDIV"]}
    values = {"N": n, "stride_we": n * k // 2, "stride_wn": k // 2, "stride_ae": n * (k // 64), "stride_an": k // 64}
    return K25_SIGNATURE, consts, values, K25_PLAN["BLOCK_N"] * K25_PLAN["KC"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kernel-dir", required=True, help="grouped-nf4-gemm's kernel/ directory")
    ap.add_argument("--config", default="k25_served", choices=("k25_served",))
    ap.add_argument("--module", default="nf4_smallm", help="the module holding the kernel (a K25 variant may differ)")
    ap.add_argument("--arch", type=int, nargs="+", default=[120, 86])
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-specialize", action="store_true", help="drop the JIT's hints (to show the inflation)")
    ap.add_argument("--fn", default="_gemm_nf4_grouped_smallm",
                    help="the jit kernel's name in --module (an ablation built to K25's signature, for instance)")
    ap.add_argument("--bytes-per-iteration", type=int, default=None,
                    help="the unroll guard for a loop with no MMA: the global-load bytes one iteration moves (a control's stream_bytes)")
    ap.add_argument("--loop-key", default="mma", choices=sorted(LOOP_KEYS),
                    help="find the inner loop by its MMAs (default) or by its weight stream (a variant with no MMA)")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.abspath(a.kernel_dir))
    import importlib
    mod = importlib.import_module(a.module)
    fn = getattr(mod, a.fn)
    report = {"config": a.config, "module": a.module, "fn": a.fn, "loop_key": a.loop_key, "plan": K25_PLAN,
              "toolchain": toolchain(), "targets": {}}
    print(f"triton {report['toolchain']['triton']}, ptxas {report['toolchain']['ptxas']}", flush=True)
    for arch in a.arch:
        for shape in K25_SHAPES:
            sig, consts, values, wpi = k25_served(shape)
            rec = compile_and_count(fn, sig, consts, values, arch=arch, num_warps=K25_PLAN["num_warps"],
                                    num_stages=K25_PLAN["num_stages"], weights_per_iteration=wpi, out_dir=a.out,
                                    tile=(16, K25_PLAN["BLOCK_N"], K25_PLAN["KC"]),
                                    tag=f"{a.module}.{a.fn}.{shape}", specialize=not a.no_specialize,
                                    loop_key=a.loop_key, bytes_per_iteration=a.bytes_per_iteration)
            report["targets"][f"sm{arch}.{shape}"] = rec
            lpw = rec.get("lane_instr_per_weight")
            body = rec["inner_loop"] and rec["inner_loop"]["instructions"]
            unroll = "unchecked" if rec.get("unroll") is None else rec["unroll"]
            loads = ", ".join(f"{op} {n:.4g}" for op, n in (rec.get("loads_per_weight") or {}).items())
            print(f"sm_{arch} {shape}: inner loop {body} instr (unroll {unroll}), "
                  f"{lpw and round(lpw, 2)} lane instr/weight; loads/weight: {loads or 'none'}", flush=True)
    if a.out:
        with open(os.path.join(a.out, "sass_count.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
