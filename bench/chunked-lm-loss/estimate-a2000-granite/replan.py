"""Plan totals (estimate + reserve + context) for the chunked-loss runs against their measured driver peaks, as loggetta's
plan_vs_driver.replan_today plans: the receipt's own hardware, workload and setup; no receipts on file, then loggetta's
committed RTX A2000 training receipts on file. E4B_CHUNKED_LM_LOSS is set as each run ran."""
import glob
import json
import os
import sys
from loggetta import Constraints, Workload, describe_model, plan
from loggetta.execution import load_observations
from loggetta.hardware import GPU, Fact, HardwareProfile, Host

GiB = 1 << 30
EV = sys.argv[1]                      # this branch's bench/chunked-lm-loss/estimate-a2000-granite
LG = sys.argv[2]                      # a loggetta checkout (its evidence/)
RUNS = [("P1", "1"), ("A6", "auto"), ("S6", "0")]
obs_dirs = sorted({os.path.dirname(f) for f in glob.glob(os.path.join(LG, "evidence", "**", "*.json"), recursive=True)})
siblings = [o for d in obs_dirs for o in load_observations(d)
            if (o.get("hardware") or {}).get("gpu", {}).get("name") == "NVIDIA RTX A2000 12GB"
            and (o.get("workload") or {}).get("kind", "train") == "train"]
topo = None
print("| run | E4B_CHUNKED_LM_LOSS | driver peak GiB | receipts on file | plan total GiB | driver / plan | reserve line | context line |")
print("|---|---|---|---|---|---|---|---|")
for name, env in RUNS:
    f = glob.glob(os.path.join(EV, "runs", name, "*.json"))[0]
    rec = json.load(open(f))
    topo = topo or describe_model(rec["model"]["model"])
    h, g = rec["hardware"], rec["hardware"]["gpu"]
    s = lambda v: Fact(v, "stated")  # noqa: E731
    cg, cw, mg, mw = g["pcie"]
    gpu = GPU(index=0, vendor="nvidia", name=g["name"], uuid=None, compute_capability=s(tuple(g["compute_capability"])),
              memory_total=s(g["memory_total"]), memory_free=s(g["memory_free"]), driver=s(g["driver"]), pcie_gen_max=s(mg),
              pcie_width_max=s(mw), pcie_gen_current=s(cg), pcie_width_current=s(cw))
    host = Host(cpu_model=s(h["cpu"]), cpus=s(h["cpus"]), memory_total=s(h["ram_limit"]), memory_available=s(h["ram_available"]),
                memory_limit=s(h["ram_limit"]))
    hw = HardwareProfile(gpus=(gpu,), host=host, platform=h["platform"])
    wl = {k: v for k, v in rec["plan"]["workload"].items() if k in Workload.__dataclass_fields__}
    os.environ["E4B_CHUNKED_LM_LOSS"] = env
    drv = rec["measured"]["driver_process_peak_bytes"]
    for label, obs in (("none", ()), ("A2000 training receipts", siblings)):
        p = plan(topo, hw, Workload(**wl), Constraints(fixed=dict(rec["setup"])), observations=obs)
        c = p.selected or (p.alternatives[0] if p.alternatives else None)
        line = lambda pre: next((ln for ln in c.lines if ln.name.startswith(pre)), None)  # noqa: E731
        res, ctx = line("allocator reserve"), line("CUDA context")
        print(f"| {name} | {env} | {drv / GiB:.3f} | {label} | {c.device_bytes / GiB:.3f} | {drv / c.device_bytes:.3f} | "
              f"{res.basis} {res.bytes / GiB:.3f} | {ctx.basis} {ctx.bytes / GiB:.3f} |")
