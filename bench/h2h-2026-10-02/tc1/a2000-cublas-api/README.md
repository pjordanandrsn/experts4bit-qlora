# What cuBLAS does inside one `torch.bmm`: fp32 against bf16 (RTX A2000, diagnostic)

The count cited in TC1 amendment 54's *Why* ([`../../../tc1/TC1-PREREG.md`](../../../tc1/TC1-PREREG.md)). It is a diagnostic that licenses
nothing, and it times nothing. Script: [`../../../tc1/bmm_api_count.py`](../../../tc1/bmm_api_count.py), run on 2026-10-06.

- [`api-torch2.8.json`](api-torch2.8.json): torch 2.8.0+cu128.
- [`api-torch2.11.json`](api-torch2.11.json): torch 2.11.0+cu128.

Both environments link cuBLAS 12.8.4 on sm_86. The host's driver (575) cannot run a cu130 build, so torch 2.12+cu130's cuBLAS 13 is not
covered. Each case runs in its own process with cuBLAS and cuBLASLt API logging on. The shape is the bucketed LoRA delta's first product,
`[12, W, 2048] @ [12, 2048, 16]`: a repeated `W`, or a new `W` on every call. The per-call counts are the difference between 80 calls
and 40, divided by 40, so process start-up drops out.

| dtype | legacy cuBLAS call per `bmm` | cuBLASLt calls per `bmm` | algorithms the heuristic returns |
|---|---|---|---|
| fp32 | `cublasSgemmStridedBatched` | `cublasLtSSSMatmulAlgoGetHeuristic`, `cublasLtSSSMatmul` | **0** |
| bf16 | `cublasGemmStridedBatchedEx` | `cublasLtTSTMatmulAlgoGetHeuristic`, `cublasLtTSTMatmul` | 21 |

The counts are the same for repeated and new shapes, and the same under both torches. Each call also issues one `cublasSetStream_v2`,
`cublasSetWorkspace_v2` and `cublasSetMathMode`, plus one `cublasGetMathMode` for fp32 and two for bf16.

What it shows: every fp32 call asks cuBLASLt's heuristic for an algorithm and gets none back, then runs anyway, on whatever cuBLAS falls
back to. Every bf16 call gets 21 candidates. A heuristic cache hit or miss is internal to cuBLASLt and does not appear in the log, so
whether a repeated shape skips work cannot be read here. Amendment 24's replay on an RTX 5090 says it does: 38 µs of host time on a
repeated fp32 shape against 119 µs on a new one.
