# RA ABBA process composition

Work item [#1363](https://github.com/pjordanandrsn/experts4bit-qlora/issues/1363).
This is a draft library proposal, with no launch CLI or proof clearance.

`ra_abba.supervise` runs old_a, new_a, new_b, old_b. Each position runs TC1
field, TC1 profile, decode (both W1/W16), capacity (warm/burst/point), quality
group 1 and quality group 12, through the adjacent existing wrappers. It pins
spec files, the staged manifest, copied venv interpreters and the common input
lock. It rechecks those bindings before and after every process. Training and
profile share a fixture; releases have separate environments and caches.
Quality reference output paths belong to the current position's fresh directory.

The controller supplies one absolute shared lock path for the box. The advisory
lock excludes cooperating RA supervisors; it does not detect foreign CUDA
contexts. A fresh output directory and exclusive process records prevent retries
from overwriting failures. A failed process, changed input or unconfirmed cleanup
stops the battery and retains the active position and completed prefix. Timeouts
leave ten minutes for retrieval and teardown; that reserve is not rental evidence.

`ra_process` retains post-kill wait failures and checks that its process group is
absent. It never treats a zero exit status with incomplete cleanup as success.
This does not cover descendants that deliberately create another process group;
the controller still needs independently collected context/process absence and
parent-death handling for nested wrappers before any next GPU process.

The current subprocess path does not yet enter the proposed verified startup
handoff. Source/publication/install/default/fallback/native receipt joins,
on-card no-skip premises, context absence, ledger/launch/deadline/retrieval and
provider teardown gates remain required. The supervisor's completion status is
`ORDERED_COMPONENTS_RECORDED_PENDING_GATES`; it never clears a release or
constructs reducer-ready feature evidence. CPU controls are composition tests.
