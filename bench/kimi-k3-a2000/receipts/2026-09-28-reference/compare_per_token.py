"""Position-by-position: the A2000 run's per-token NLL against Fireworks' prompt logprobs.

Both score the same 90-token paragraph (token strings are checked, not assumed), and
position 0 has no prediction in either, so 89 positions are compared. Fireworks returns
logprobs at reduced precision (values like -8.75, -2.625), so single-position
differences under ~0.01 nats are below its reporting resolution.
"""
import json
import math
import pathlib
import statistics
import sys

ours = json.loads(pathlib.Path(sys.argv[1]).read_text())
ref = json.loads(pathlib.Path(sys.argv[2]).read_text())
tok_strings = json.loads(pathlib.Path(sys.argv[3]).read_text())   # our ids decoded, one per id
rt = [t for t, _ in ref["ppl"]["per_token"]]
if tok_strings != rt:
    bad = [i for i, (a, b) in enumerate(zip(tok_strings, rt)) if a != b]
    raise SystemExit(f"token strings differ (len {len(tok_strings)} vs {len(rt)}; first bad {bad[:5]})")
a = ours["per_token_nll"]                                  # NLL of token t+1, t = 0..88
b = [-lp for _, lp in ref["ppl"]["per_token"][1:]]         # same predictions
assert len(a) == len(b) == 89, (len(a), len(b))
d = [x - y for x, y in zip(a, b)]
ma, mb = statistics.fmean(a), statistics.fmean(b)
cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
r = cov / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
worst = sorted(range(89), key=lambda i: -abs(d[i]))[:5]
out = {"positions": 89, "ours_nll": ma, "ref_nll": mb, "ours_ppl": math.exp(ma),
       "ref_ppl": math.exp(mb), "mean_delta": statistics.fmean(d),
       "mean_abs_delta": statistics.fmean(abs(x) for x in d),
       "median_abs_delta": statistics.median(abs(x) for x in d),
       "pearson_r": r,
       "within_0.05": sum(abs(x) <= 0.05 for x in d), "within_0.2": sum(abs(x) <= 0.2 for x in d),
       "worst": [{"pos": i + 1, "token": rt[i + 1], "ours": a[i], "ref": b[i], "delta": d[i]}
                 for i in worst]}
print(json.dumps(out, indent=1))
pathlib.Path("per_token_compare.json").write_text(json.dumps(out, indent=1))
