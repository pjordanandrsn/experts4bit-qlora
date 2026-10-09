# RA Linux child lifetime

Work item [#1363](https://github.com/pjordanandrsn/experts4bit-qlora/issues/1363).
Draft control proposal; no rental or GPU clearance.

On Linux, each `ra_process.run` enters the adjacent `ra_child_guard.py` through
the selected interpreter's `-I -S -B` startup. It verifies its expected parent,
arms `PR_SET_PDEATHSIG` with SIGKILL, reads the signal back, checks its parent
again and writes exclusive guard evidence before exec of the original argv.
Missing or changed evidence, including bootstrap source drift, fails the component.
Timeouts retain guard verification without masking the original failure. Launches must originate on
the parent main thread because Linux ties this signal to the spawning thread.
The guard uses no fork-time Python callback or unverified site startup.

This covers cooperating RA launches even when nested wrappers create separate
sessions. The original target retains its existing startup policy; this bootstrap
does not implement the proposed verified install/import handoff. The selected
interpreter and stdlib still require independent installation verification.
Linux clears parent-death signals for a child's own forks; arbitrary descendants,
credential changes and external CUDA contexts remain separate gates. Setuid and
setgid targets and file capabilities refuse. Group absence alone is not GPU-context absence.

Native Linux CPU controls must retain parent, child and grandchild identities,
read guard receipts before terminating the owned parent, and reap the descendants.
No foreign process or GPU context is part of those controls. Non-Linux CPU runs
record that the Linux guard was not engaged and cannot satisfy an on-card gate.

Validation: 27 guard mutation/receipt controls plus 46 training/ABBA controls
passed without skips. The pinned extracted Linux CPython 3.11.13 preserved the
signal across exec, retained verified timeout evidence, refused a wrong-parent
mutant before target code, and killed/reaped two nested workers in separate
sessions after their owned parent died. This used native host libc, no full-image
installation or GPU. Verified install/import startup and external context absence
remain pending.
