"""Diagnostic count (no timing; TC1 amendment 54's Why): which cuBLAS / cuBLASLt API calls one torch.bmm makes, and how many
algorithms cuBLASLt's heuristic returns, by dtype, with a repeated shape against fresh shapes. Run once per (dtype, pattern) in its own process with cuBLAS / cuBLASLt API logging pointed at a file; the parent
counts the log's entries per call. Shapes mirror the bucketed LoRA delta's products: [G, W, K] @ [G, K, r] with r 16."""
import os, sys, subprocess, json, re, collections
if len(sys.argv) > 1 and sys.argv[1] == "child":
    import torch
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[2]]
    pattern, n = sys.argv[3], int(sys.argv[4])
    G, K, R = 12, 2048, 16
    torch.cuda.init()
    warm = torch.bmm(torch.randn(G, 64, K, device="cuda", dtype=dt), torch.randn(G, K, R, device="cuda", dtype=dt))
    torch.cuda.synchronize()
    print("MARK-START", flush=True)
    for i in range(n):
        W = 256 if pattern == "repeat" else 256 + 8 * (i + 1)          # fresh: a new row count every call
        torch.bmm(torch.randn(G, W, K, device="cuda", dtype=dt), torch.randn(G, K, R, device="cuda", dtype=dt))
    torch.cuda.synchronize()
    sys.exit(0)
out = {"rows": []}
import torch
out["torch"] = torch.__version__
for dt in ("fp32", "bf16"):
    for pattern in ("repeat", "fresh"):
        logf = f"/tmp/cublas_{dt}_{pattern}.log"; logl = f"/tmp/cublaslt_{dt}_{pattern}.log"
        env = dict(os.environ, CUBLAS_LOGINFO_DBG="1", CUBLAS_LOGDEST_DBG=logf, CUBLASLT_LOG_LEVEL="5", CUBLASLT_LOG_FILE=logl)
        def api_counts(path, lt):
            c = collections.Counter()
            if os.path.exists(path):
                text = open(path, errors="replace").read()
                pat = r"\[Api\]\[(\w+)\]" if lt else r"function \S+ (\w+)\("
                c.update(re.findall(pat, text))
                if lt:
                    c.update("heuristicResults=" + h for h in re.findall(r"heuristicResults=\[(\d+)\]", text))
            return c
        per = {}
        for n in (40, 80):
            for f in (logf, logl):
                if os.path.exists(f): os.remove(f)
            subprocess.run([sys.executable, __file__, "child", dt, pattern, str(n)], env=env, check=True, capture_output=True)
            per[n] = (api_counts(logf, False), api_counts(logl, True))
        diff = lambda i: {k: (per[80][i][k] - per[40][i][k]) / 40 for k in set(per[80][i]) | set(per[40][i]) if per[80][i][k] != per[40][i][k]}
        row = {"dtype": dt, "pattern": pattern, "cublas_per_bmm": diff(0), "cublaslt_per_bmm": diff(1),
               "cublas_lines_n40": sum(per[40][0].values()), "cublaslt_lines_n40": sum(per[40][1].values())}
        out["rows"].append(row); print(json.dumps(row), flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/bmm_api_count.json", "w"), indent=1)
