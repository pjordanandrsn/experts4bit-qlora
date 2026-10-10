# DQ11 CPU policy check before quotes

Operational defence before the existing quote/rental controller, under
[Amendment 4](DQ11-AMENDMENT-4.md). This adds no model work or GPU computation
and does not change the scientific arm, training loop, quality/headroom gates,
package lock, budget or driver/device requirements.

`dq11_arm.prepare` explicitly sets both `torch.backends.cuda.matmul.allow_tf32`
and `torch.backends.cudnn.allow_tf32` to false after library imports. That is
the harness policy; plain Torch defaults are not the reference. The CPU probe
parses the actual sealed `prepare` source and executes those exact two false
assignments. A missing, duplicate or changed assignment refuses. The arm
source remains unchanged.

Before any read-only quote or guarded rental, `dq11_launch` executes a fresh
policy probe and requires its result alongside the original all-101-URL gate.
The probe requires Linux x86_64 / Python 3.11, the full 101-package science
lock, actual Torch `2.12.1+cu130` / CUDA build `13.0`, hash-authorized loaded
Torch source and the installed e4b/Loggetta source commits. Three fresh CPU
processes import the actual L/U/U0 libraries, record flags before and after
the sealed TF32 assignments, and require the registered shipped policy.

U/U0 use the locked Unsloth's explicit `UNSLOTH_ALLOW_CPU=1` import support only
inside isolated CPU probe children, with the original compile-disable and
return-logits settings. Every result labels that support and its limitation.
The probe requires CUDA unavailable and uninitialized. It loads no model,
runs no forward/backward and certifies no CUDA kernel or GPU-path behavior.
The scientific arms still validate their actual prepared policy before
training; the CPU flag check is additional defence, not a substitute.

The launch controller supplies a new nonce for each invocation and binds the
returned actual flags, source commits, wheel versions and science/arm/probe
hashes to its current launch source. Missing imports, timeouts, malformed or
replayed results and mismatches refuse before quote/rental. Raw stdout/stderr
and failure evidence remain in the external gate record. A previous receipt
cannot replace a fresh probe.

Run the controller in the exact installed science CPU environment when it is
available locally. An operator on another platform can set
`DQ11_POLICY_COMMAND` to a JSON argument vector for a trusted transport to that
CPU environment. The command must execute this same sealed probe at the bound
source and forward the controller's `--source` and `--nonce` arguments. The
controller validates the response itself; an absent transport defaults to a
fresh local Python child and refuses if its exact runtime is unavailable.
Neither command strings nor shell evaluation are used by the controller.

Use `--gate-only` for the pre-quote check, and execute the controller again for
the final bound launch. Review this change and exercise its actual exact-wheel
gate, then run the fifth zero-rental rehearsal from the merged source containing
both Amendment 4 and this gate. The measured supplement and separate maintainer
draw-4 ACK remain required. This document authorizes no rental.
