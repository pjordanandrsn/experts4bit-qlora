### `E4B_NF4_SMALLM_BW` defaults on for Mixtral at 2 to 4 tokens (P234, LICENSED; #1313)

Lane P234 read `E4B_NF4_SMALLM_BW=1` on Mixtral-8x7B's default NF4 server (`bench/p234/RESULTS-p234.md`): on one RTX
5090, two requests decode 2.12452x and four 1.76481x as fast, within P110's quality bars at group 4.
- **The default:** an unset `E4B_NF4_SMALLM_BW` resolves at engine build, after the levers and before any graph is
  captured. On `mixtral` it routes the NF4 store's calls of 2 to 4 tokens (buckets 2 and 4) through K33's per-row
  bandwidth GEMV; everywhere else it is off. The resolution and its reason are in `nf4_smallm_bw["source"]`
  (`explicit`, `default-allowlisted`, or `default-off: <why>`).
- **Served tokens change** at two and four requests on Mixtral: each row now runs K33's arithmetic instead of K25's
  (within the kernels' tolerance contract; the quality rung read it). One request is unchanged.
  `E4B_NF4_SMALLM_BW=0` is the way back.
- **Never a refusal by default:** wherever an explicit `1` would refuse (an int4 or MXFP4 store,
  `E4B_NF4_GROUPED_SMALLM=1`, a layer off the all-resident collapse, a kernel package without K33, an expert shape
  where K33 does not engage on the device), the default resolves off and names the reason. At call time it yields to an
  int4 or MXFP4 store and to `E4B_NF4_GROUPED_SMALLM=1` without raising. An explicit `1` keeps its refusals and still
  routes 2 to 8 tokens on any family.
- **Scope: the RTX 5090 only.** The default is the read's scope, not the kernel's: it engages on sm_120 at 170 SMs,
  the part P234 read. Any other part, a 188-SM sm_120 included where K33 also engages, resolves off with the part
  named (`default-off: no P234 read on this part`) and stays opt-in until a read of its own. On the 5090 it also needs
  K33 to engage (Mixtral-8x7B's two expert shapes); other Mixtral checkpoints' shapes resolve off. Calls of 5 to 8
  tokens (bucket 8) stay opt-in.
- **One engine per process:** the resolved default is process-wide; each engine build replaces it.

Files: `experts4bit_qlora/engines/hot_residency.py`, `experts4bit_qlora/serve_paged.py`,
`tests/test_nf4_smallm_bw_default.py`, `tests/test_nf4_smallm_bw.py`, `docs/SERVING.md`, `docs/STATUS.md`.
