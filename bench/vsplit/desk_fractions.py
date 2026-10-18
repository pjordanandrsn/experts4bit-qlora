"""The verify split's registered predictions: P132's desk prediction v2 (bench/p132/desk_out.json, accepted as final
2026-10-11) restated as FRACTIONS of the increment I(n), so they carry to any card. $0; reads only desk_out.json.

The desk's millisecond values belong to sd2-5090-3's card; their ratios to that card's I(n) are what this lane predicts
for the card it runs on. Per n (lo / central / hi):
  bytes / I            the extra distinct experts' byte cost (one value)
  route / I            the expert path's routing growth (P132's lo / central / hi)
  dense / I            the dense projections' growth
  attn_norm / I        attention and norms per extra row (one value)
  share_under_H        H's reading if the hole lies in the expert path: (I - bytes - dense - attn_norm) / (I - bytes);
                       lo takes dense hi, hi takes dense lo
  S_over_need_under_H  S(n) / need(n) with S from the desk's S_under_H; BUILD at n iff >= 1
They are REPORTED beside the reading; H and BUILD are decided by the reducer's rule, not by these numbers.

  python bench/vsplit/desk_fractions.py --desk bench/p132/desk_out.json --out bench/vsplit/desk_fractions.json
"""
import argparse
import json
from pathlib import Path

NS = ("2", "3", "4")


def fractions(desk: dict) -> dict:
    out = {}
    for n in NS:
        p = desk["per_n"][n]
        inc, by, an = p["I"], p["bytes"], p["attn_norm"]
        d_lo, d_c, d_hi = p["dense"]
        need = desk["need"][n]["threshold"]
        out[n] = {
            "I_ms_on_sd2_card": inc,
            "bytes": by / inc,
            "route": [r / inc for r in p["route"]],
            "dense": [d / inc for d in p["dense"]],
            "attn_norm": an / inc,
            "share_under_H": [(inc - by - d_hi - an) / (inc - by), (inc - by - d_c - an) / (inc - by),
                              (inc - by - d_lo - an) / (inc - by)],
            "S_over_need_under_H": [s / need for s in p["S_under_H"]],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desk", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fr = fractions(json.loads(Path(a.desk).read_text(encoding="utf-8")))
    Path(a.out).write_bytes((json.dumps(fr, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    f3 = lambda t: "/".join(f"{x:.3f}" for x in t)  # noqa: E731
    print(f"{'n':>2} | {'bytes/I':>7} | {'route/I lo/c/hi':>17} | {'dense/I lo/c/hi':>17} | {'attn+norm/I':>11} |"
          f" {'share under H lo/c/hi':>21} | {'S/need under H':>17}")
    for n in NS:
        r = fr[n]
        print(f"{n:>2} | {r['bytes']:7.3f} | {f3(r['route']):>17} | {f3(r['dense']):>17} | {r['attn_norm']:11.3f} |"
              f" {f3(r['share_under_H']):>21} | {f3(r['S_over_need_under_H']):>17}")


if __name__ == "__main__":
    main()
