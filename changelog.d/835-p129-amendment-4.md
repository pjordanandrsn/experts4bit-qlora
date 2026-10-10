### P129 Amendment 4 (#835): DEFAULT_ON's second-host box reruns on current main

`tc1-5090-148` measured nothing. Its container restarted three minutes in, and the driver at the pinned commit counted its own ssh
shell as the lane, so the box waited until stopped. Main's driver now checks lane identity and host uptime (#1512, #1517).

The rerun pins e4b main and keeps grouped-nf4-gemm at `d1f64ba`. A package-diff audit against box 147's build lists seven changed files,
all off the Qwen3-30B-A3B training path, and the generator refuses any other. It avoids machines 27708 and 46990. The lane has spent
$6.653 of $15.
