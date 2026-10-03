"""Compare arm outputs (graph-replay logits per decode step) of two check_multibucket runs."""
import sys
import torch
a, b = (torch.load(p) for p in sys.argv[1:3])
for k in ("g", "e"):
    worst, agree, n = 0.0, 0, 0
    for (ra, xa), (rb, xb) in zip(a[k], b[k]):
        assert ra == rb
        worst = max(worst, (xa - xb).abs().max().item())
        agree += int((xa.argmax(-1) == xb.argmax(-1)).sum()); n += xa.shape[0]
    print(f"COMPARE {sys.argv[1]} vs {sys.argv[2]} [{k}]: max abs logit diff {worst:.3e}, argmax agree {agree}/{n}")
