"""#1526 training half: the registered comparison over the four L4 receipts (e4b 29370aaa)."""
import json
import os
R = os.path.dirname(os.path.abspath(__file__))
MiB, GiB = 1 << 20, 1 << 30
H = 2048
MAP = {"frozen expert stacks": ["frozen expert stacks"], "attention projections (NF4)": ["attention projections (NF4)"],
       "fused q/k/v training projection (fp32 absmax)": ["fused q/k/v training projection"],
       "dense weights (bf16)": ["dense weights (bf16)"], "expert LoRA adapters": ["expert LoRA adapters"],
       "attention LoRA adapters": ["attention LoRA adapters"], "optimizer state (adamw_8bit)": ["optimizer state"]}
out = {}
for T in (4096, 1024):
    for ad in ("fp32", "native"):
        r = json.load(open(f"{R}/L4-T{T}-{ad}.json"))
        P = max(r["peak_allocated_bytes_per_step"][1:])
        est = r["e4b_estimate_attention_dtype_corrected"]
        items = {i["name"]: i["bytes"] for i in est["items"] if i["where"] == "device"}
        ev = items.get("held-out evaluation, above the training step", 0)
        E_step = est["device_bytes"] - ev
        c = r["census"]
        print(f"== T={T} {ad}: P {P / MiB:.1f} MiB, E_step {E_step / MiB:.1f} MiB, E_step/P {E_step / P:.4f}, "
              f"eval item {ev / MiB:.1f} MiB; grad dtypes {c['grad_dtypes']}; unclassified after loop "
              f"{c['unclassified_allocated'] / MiB:.1f} MiB")
        statics_meas = 0
        if T == 4096:
            print("   statics: item | estimate MiB | census MiB | residual MiB | % of P")
        for name, keys in MAP.items():
            e = items.get(name, 0)
            m = sum(c["items_bytes"].get(k, 0) for k in keys)
            statics_meas += m
            if T == 4096:
                flag = "  <-- >1% of P" if abs(m - e) > 0.01 * P else ""
                print(f"     {name:46s} {e / MiB:9.1f} {m / MiB:9.1f} {(m - e) / MiB:9.2f} {100 * (m - e) / P:6.2f}%{flag}")
        g_e, g_m = items.get("adapter gradients", 0), c["grad_bytes_from_params"]
        if T == 4096:
            print(f"     {'adapter gradients':46s} {g_e / MiB:9.1f} {g_m / MiB:9.1f} {(g_m - g_e) / MiB:9.2f} {100 * (g_m - g_e) / P:6.2f}%")
        held = statics_meas + g_m + c["unclassified_allocated"]
        transient = P - held
        bound = 4 * T * H * 2
        act = items["activations"]
        branch_m, branch_e = transient - bound, act - bound
        print(f"   transient {transient / MiB:.1f} MiB = saved inputs {bound / MiB:.1f} + branch {branch_m / MiB:.1f}; "
              f"estimate activations {act / MiB:.1f} (branch {branch_e / MiB:.1f}); RESIDUAL branch {(branch_m - branch_e) / MiB:+.1f} MiB")
        out[(T, ad)] = {"branch_m": branch_m, "branch_e": branch_e, "P": P, "E_step": E_step}
json.dump({f"{k[0]}-{k[1]}": v for k, v in out.items()}, open(os.path.join(R, "l4-branches.json"), "w"), indent=1)
