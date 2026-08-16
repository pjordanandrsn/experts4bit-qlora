# Offline placement sweep: which DRAM budget lands NVMe traffic mass in
# the G4 band (10-15%)? Pure solver, no model load. Run on the box.
import sys

sys.path[:0] = ["/root/e4b", "/root/gnf4", "/root/gnf4/kernel"]

from nvme_arena import load_index  # noqa: E402

from experts4bit_qlora import solve_placement  # noqa: E402

idx = load_index("/root/q235.arena")
print(f"row_stride={idx['row_stride']}")
print("dram_gb  vram_frac  dram_frac  nvme_frac")
for gb in (4, 6, 8, 10, 12, 15, 20, 25, 30, 40):
    m = solve_placement(
        n_layers=94, n_experts=128, bytes_per_expert=idx["row_stride"],
        vram_budget_bytes=int(10.85 * 2**30),
        dram_budget_bytes=gb * 2**30,
        calibration="/root/out/calib.json",
        profile_path="/root/out/route_profile.jsonl")
    ma = m["masses"]
    print(f"{gb:7d}  {ma['vram_frac']:.4f}     {ma['dram_frac']:.4f}"
          f"     {ma['nvme_frac']:.4f}")
