"""Best-effort torch.profiler arm for the H-D branch (prereg_baremetal4 R5).

Profiles training steps 2-4 inside the REAL trainer loop and dumps
key_averages + a cudaMemcpy/idle summary to E4B_PROF_OUT. Identification,
not pass/fail.
"""
import os

import torch
from transformers import TrainerCallback

from axolotl.integrations.base import BasePlugin


class _ProfCallback(TrainerCallback):
    def __init__(self):
        self.prof = None

    def on_step_begin(self, args, state, control, **kw):
        if state.global_step == 2 and self.prof is None:
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
            )
            self.prof.__enter__()

    def on_step_end(self, args, state, control, **kw):
        if state.global_step == 4 and self.prof is not None:
            self.prof.__exit__(None, None, None)
            out = os.environ.get("E4B_PROF_OUT", "/home/ubuntu/dq/prof_r2048.txt")
            evs = self.prof.key_averages()
            with open(out, "w") as f:
                f.write(evs.table(sort_by="self_cuda_time_total", row_limit=48))
                f.write("\n\n== copy + sync ops ==\n")
                for e in evs:
                    n = e.key.lower()
                    if "memcpy" in n or "synchronize" in n or "cudastream" in n:
                        dev_us = getattr(e, "self_device_time_total", 0) or getattr(
                            e, "self_cuda_time_total", 0
                        )
                        f.write(
                            f"{e.key}: count={e.count} cpu_total_us={e.self_cpu_time_total:.0f} "
                            f"dev_total_us={dev_us:.0f}\n"
                        )
            print(f"E4B_PROF: wrote {out}", flush=True)
            self.prof = None
            control.should_training_stop = True
        return control


class ProfilerPlugin(BasePlugin):
    def add_callbacks_post_trainer(self, cfg, trainer):
        return [_ProfCallback()]
