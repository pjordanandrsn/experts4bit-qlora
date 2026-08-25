# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Parse a torch-profiler key-averages table (the `--host-brackets`
kernels.txt step_decomp already writes) into a per-step DEVICE budget.

Three ways this table lies if summed naively, all guarded here:
1. **Region rows** (`e4b::*`, `ProfilerStep*`) are walls including
   children and host gaps -- the 2026-06 occupancy-parser defect. Their
   Self CUDA % exceeds 100% because they are not part of the kernel
   total. Excluded.
2. **Two views are interleaved.** `aten::topk` (op) and
   `sbtopk::gatherTopK` (its kernel) BOTH carry self-CUDA; summing
   across views double-counts. Only the KERNEL view is summed -- it is
   what the device actually ran.
3. **The table is row-limited**, so a kernel view can silently cover a
   fraction of the work. The footer's `Self CUDA time total` is the
   ground truth; coverage below MIN_COVERAGE REFUSES rather than
   publishing a partial budget as if it were the step.

Column note: Self CUDA is column 7, NOT column 9 (CUDA total) --
dispatcher rows (`aten::matmul`, `aten::linear`, `aten::to`) inherit a
CUDA-total from their children with zero self, and summing column 9
read 23.6 ms/step against a 12.6 ms truth."""

import argparse
import json
import re
import sys

REGION_RE = re.compile(r"^\s*(e4b::|ProfilerStep)")
HDR_RE = re.compile(r"active window:\s*(\d+)\s*/")
# Self CUDA is column 6 of the key-averages table; rows are name-then-cols.
ROW_RE = re.compile(r"^\s*(.+?)\s{2,}([\d.]+)%\s+([\d.]+\w*s)\s+([\d.]+)%\s+"
                    r"([\d.]+\w*s)\s+([\d.]+\w*s)\s+([\d.]+\w*s)\s+"
                    r"([\d.]+)%\s+([\d.]+\w*s)\s+([\d.]+\w*s)\s+(\d+)\s*$")

MIN_COVERAGE = 0.90
TOTAL_RE = re.compile(r"Self CUDA time total:\s*([\d.]+\s*\w+)")
# Rows that are NOT device kernels: op-view dispatch, user regions, and
# profiler/runtime bookkeeping.
NON_KERNEL = ("aten::", "e4b::", "ProfilerStep", "cudaLaunch",
              "cuLaunchKernel", "Activity Buffer", "cudaMemcpy",
              "cudaStreamSynchronize", "cudaDeviceSynchronize",
              "cudaFree", "cudaMalloc", "Runtime Trigger")
ELEMENTWISE = ("elementwise_kernel", "unrolled_elementwise",
               "vectorized_elementwise", "reduce_kernel", "softmax",
               "CatArrayBatched", "indexSelect", "indexFunc",
               "copy_device_to_device")
MATMUL = ("gemv", "gemm", "mm", "sm80", "sm90", "sm100", "cutlass",
          "ampere", "hopper", "blackwell", "nf4")


def _us(tok):
    """'1.234ms' / '567.000us' / '1.2s' -> microseconds."""
    m = re.match(r"([\d.]+)\s*(\w+)", tok)
    v, u = float(m.group(1)), m.group(2)
    return v * {"us": 1.0, "ms": 1e3, "s": 1e6, "ns": 1e-3}[u]


def is_kernel(name):
    return not name.startswith(NON_KERNEL)


def classify(name):
    low = name.lower()
    if "topk" in low or "sort" in low:
        return "router"
    if any(k in low for k in MATMUL):
        return "matmul"
    if any(k in low for k in ELEMENTWISE):
        return "elementwise"
    if low.startswith("memcpy") or "memset" in low:
        return "memcpy"
    return "other"


def parse(path):
    text = open(path).read()
    m = HDR_RE.search(text)
    if not m:
        raise SystemExit("no 'active window: N/M' header -- not a "
                         "step_decomp kernels.txt")
    steps = int(m.group(1))
    if steps <= 0:
        raise SystemExit(f"non-positive active step count {steps}")
    mt = TOTAL_RE.search(text)
    if not mt:
        raise SystemExit("no 'Self CUDA time total' footer -- cannot "
                         "verify coverage, refusing to report a budget")
    cuda_total_us = _us(mt.group(1).replace(" ", ""))
    rows, seen = [], set()
    for line in text.splitlines():
        if REGION_RE.match(line):
            continue                      # region wall, not a kernel
        m = ROW_RE.match(line)
        if not m:
            continue
        # Columns after Name: SelfCPU%, SelfCPU, CPUtot%, CPUtot,
        # CPUavg, SELF CUDA(7), SelfCUDA%, CUDAtot(9), CUDAavg, Calls.
        # Group 7, NOT 9: dispatcher rows (aten::matmul, aten::linear,
        # aten::to) carry a CUDA-total inherited from their children and
        # a Self CUDA of zero -- summing column 9 double-counts every
        # nested level (it read 23.6 ms/step against a 13.3 ms truth).
        name, self_cuda, calls = m.group(1).strip(), _us(m.group(7)), \
            int(m.group(11))
        if self_cuda <= 0 or name in seen:   # zero-self dispatcher rows
            continue
        seen.add(name)
        if not is_kernel(name):              # op view -- would double-count
            continue
        rows.append({"name": name, "us_per_step": self_cuda / steps,
                     "calls_per_step": calls / steps,
                     "kind": classify(name)})
    return steps, rows, cuda_total_us


def budget(steps, rows, cuda_total_us):
    by = {}
    for r in rows:
        b = by.setdefault(r["kind"], {"us_per_step": 0.0,
                                      "calls_per_step": 0.0})
        b["us_per_step"] += r["us_per_step"]
        b["calls_per_step"] += r["calls_per_step"]
    total = sum(b["us_per_step"] for b in by.values())
    truth = cuda_total_us / steps
    cov = total / truth if truth > 0 else 0.0
    return {"active_steps": steps, "total_device_us_per_step": total,
            "device_us_per_step_truth": truth, "coverage": cov,
            "coverage_ok": cov >= MIN_COVERAGE,
            "by_kind": by,
            "rows": sorted(rows, key=lambda r: -r["us_per_step"])}


def self_test():
    import tempfile

    def w(t):
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        f.write(t)
        f.close()
        return f.name

    hdr = ("profiled decode steps: 127 (active window: 10/10)\n"
           "  Name  Self CPU %  Self CPU  CPU total %  CPU total  "
           "CPU time avg  Self CUDA  Self CUDA %  CUDA total  "
           "CUDA time avg  # of Calls\n")
    # Self CUDA (col 7) and CUDA total (col 9) DIFFER on every row, so a
    # column mix-up fails. aten::topk is the op view of gatherTopK: both
    # carry self-CUDA and only the kernel may be summed. aten::matmul is
    # a zero-self dispatcher. e4b::attn is a region wall whose self
    # exceeds the total.
    body = (
        "                e4b::attn         0.00%       0.000us         "
        "0.00%       0.000us       0.000us     279.973ms       185.91%  "
        "   279.973ms     486.065us           576\n"
        "             aten::matmul         0.69%       6.061ms         "
        "5.21%      45.676ms      15.729us       0.000us         0.00%  "
        "    50.000ms       7.974us          1000\n"
        "               aten::topk         0.65%       5.737ms         "
        "1.35%      11.864ms      20.598us      30.000ms         2.54%  "
        "    38.000ms       6.656us           500\n"
        "        _gemv_nf4_grouped         0.00%       0.000us         "
        "0.00%       0.000us       0.000us      50.000ms        38.58%  "
        "    60.000ms      50.435us          1000\n"
        "void at::native::elementwise_kernel<128, 4, at::nati...      "
        "   0.00%       0.000us         0.00%       0.000us       "
        "0.000us      10.000ms        10.42%      70.000ms       "
        "1.177us          5000\n"
        "void at::native::sbtopk::gatherTopK<float, unsigned ...      "
        "   0.00%       0.000us         0.00%       0.000us       "
        "0.000us      20.000ms         1.40%      21.000ms       "
        "3.650us           500\n"
        "         cuLaunchKernelEx         0.10%       1.000ms         "
        "0.10%       1.000ms       1.000us       5.000ms         0.01%  "
        "     5.000ms       0.100us          1000\n")

    steps, rows, tot = parse(w(hdr + body + "Self CUDA time total: "
                               "80.000ms\n"))
    assert steps == 10, steps
    names = {r["name"] for r in rows}
    assert not any(n.startswith(("e4b::", "aten::")) for n in names), names
    assert not any("cuLaunchKernel" in n for n in names), names
    b = budget(steps, rows, tot)
    # kernel view only: gemv 50 + elementwise 10 + gatherTopK 20 = 80 ms
    # over 10 steps = 8000 us/step, and coverage is exactly 1.0.
    assert abs(b["by_kind"]["matmul"]["us_per_step"] - 5000.0) < 1e-6, \
        b["by_kind"]
    assert abs(b["by_kind"]["elementwise"]["us_per_step"] - 1000.0) < 1e-6
    assert abs(b["by_kind"]["router"]["us_per_step"] - 2000.0) < 1e-6
    assert abs(b["total_device_us_per_step"] - 8000.0) < 1e-6
    assert b["coverage_ok"] and abs(b["coverage"] - 1.0) < 1e-9, b["coverage"]

    # a row-limited table must REFUSE, not publish a partial budget
    steps2, rows2, tot2 = parse(w(hdr + body
                                  + "Self CUDA time total: 160.000ms\n"))
    b2 = budget(steps2, rows2, tot2)
    assert not b2["coverage_ok"] and abs(b2["coverage"] - 0.5) < 1e-9, \
        b2["coverage"]

    # missing footer must refuse outright
    try:
        parse(w(hdr + body))
    except SystemExit:
        pass
    else:
        raise AssertionError("missing footer must refuse")
    print("self-test PASS (region exclusion, op-vs-kernel view, column 7 "
          "not 9, runtime rows, unit scaling, coverage gate, footer "
          "refusal)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernels_txt", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.kernels_txt:
        sys.exit("need a kernels.txt (or --self-test)")
    rep = budget(*parse(a.kernels_txt))
    if a.out:
        open(a.out, "w").write(json.dumps(rep, indent=1))
    if not rep["coverage_ok"]:
        sys.exit(
            f"REFUSE: the kernel view covers {rep['coverage'] * 100:.1f}% "
            f"of the profiler's Self CUDA total "
            f"({rep['device_us_per_step_truth'] / 1000:.2f} ms/step), "
            f"below {MIN_COVERAGE * 100:.0f}%. The table is row-limited -- "
            f"re-profile with a larger row_limit before quoting a budget.")
    t = rep["total_device_us_per_step"]
    print(f"device budget: {t/1000:.2f} ms/step over "
          f"{rep['active_steps']} active steps (coverage "
          f"{rep['coverage']*100:.1f}% of "
          f"{rep['device_us_per_step_truth']/1000:.2f} ms truth)")
    for k, v in sorted(rep["by_kind"].items(),
                       key=lambda kv: -kv[1]["us_per_step"]):
        print(f"  {k:12s} {v['us_per_step']/1000:6.2f} ms  "
              f"{v['calls_per_step']:7.0f} launches/step  "
              f"{v['us_per_step']/t*100:5.1f}%")
    print("  top rows:")
    for r in rep["rows"][:10]:
        print(f"    {r['name'][:52]:54s} {r['us_per_step']:8.1f} us "
              f"x{r['calls_per_step']:.0f}  [{r['kind']}]")


if __name__ == "__main__":
    main()
