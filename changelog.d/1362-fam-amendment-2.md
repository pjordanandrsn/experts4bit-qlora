### FAM Amendment 2 (#1362): the box scores the largest shape first

`fam-qw36-1` ($1.639) VOIDed on a harness bug. On the hybrid Qwen3.6, the first runner (shape 1, padded) sized and froze
the model's linear-state pool at 17 slots, and e4b refused the 28 that shape 12 needs. `fam_box.run_cells` now scores
shape 12 first. `tests/test_fam_shape_order.py` reproduces the refusal at the box's slot counts and shows the fix
leaves the arithmetic unchanged: a shape-1 pass is bit-identical on either pool size, and after a shape-12 pass. The
rule, gates and predictions are unchanged. The registration's one rerun, `fam-qw36-2`, follows the merge.
