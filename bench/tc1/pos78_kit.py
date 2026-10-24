"""TC1 amendment 78's launch caps and offer screen (TC1-PREREG.md, amendment 78, "Budget").

The kit's market watch admits an offer only through ``offer_why``, and the launch template
(``pos78_manifest_template.json``) carries the same caps, so a dearer offer is refused in both
places: the screen never pins it, and the launcher drops any offer that bills download above the
template's ``max_download_usd_per_gb``. Pure Python, no network: the caller reads the offer's
hourly rate (GPU plus the 220 GB disk) and its download price from the market.
"""
from __future__ import annotations

RATE_USD_H = 1.00            # GPU plus 220 GB of storage, the declared box rate
DL_USD_GB = 0.02             # the download ceiling: the lane fits the 15-dollar rail (maintainer ruling)
GUARD_H = 1.5
DOWNLOAD_GB = 100
DISK_GB = 220
GPU_RAM_MIB = (24000, 24600)  # the listing's gpu_ram: a 24 GiB card (the GPU name alone admits a 48 GB "RTX 4090")
SSH_READY_S = 420            # bench/common rule 5 on Vast
CAP_STARTED = 3              # the most boxes that may START
RETRY_IDS = 8                # tc1-4090-r1 .. r8
# A box that does not start: 600 s load + 420 s ssh + 180 s of pre-flight probe windows + 120 s staging
# + 120 s refusal, evidence and destroy = 1,440 s at the box rate, plus the probes' 1.17 GB download.
NONSTART_S = 600 + SSH_READY_S + 4 * 45 + 120 + 120
PROBE_GB = 1.17


def offer_why(gpu_ram_mib, hourly_usd, download_usd_per_gb, num_gpus=1):
    """Why the screen refuses this offer, or "" when it admits it. A missing price refuses."""
    if int(num_gpus or 0) != 1:
        return f"{num_gpus} GPUs, not one"
    if gpu_ram_mib is None or not GPU_RAM_MIB[0] <= float(gpu_ram_mib) <= GPU_RAM_MIB[1]:
        return f"gpu_ram {gpu_ram_mib} MiB is not a 24 GiB card's {GPU_RAM_MIB[0]}-{GPU_RAM_MIB[1]}"
    if hourly_usd is None or float(hourly_usd) > RATE_USD_H:
        return f"{hourly_usd} USD/h (GPU plus {DISK_GB} GB) is over {RATE_USD_H:.2f}"
    if download_usd_per_gb is None or float(download_usd_per_gb) > DL_USD_GB:
        return f"download {download_usd_per_gb} USD/GB is over {DL_USD_GB:.2f}"
    return ""


def box_worst():
    """A started box: the guard at the box rate plus the download allowance at the ceiling."""
    return round(GUARD_H * RATE_USD_H + DOWNLOAD_GB * DL_USD_GB, 2)


def nonstart_worst():
    """A box that does not start, rounded UP to the cent."""
    cents = (NONSTART_S / 3600 * RATE_USD_H + PROBE_GB * DL_USD_GB) * 100
    return (int(cents) + (cents > int(cents))) / 100


def lane_worst():
    """bench/common rule 4: the cap's started boxes at the box worst, the retry ids at the non-start worst."""
    return round(CAP_STARTED * box_worst() + RETRY_IDS * nonstart_worst(), 2)
