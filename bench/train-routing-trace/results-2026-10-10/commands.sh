# e4b @ 450b07fff4472bfe26789fa8bc3ff0a354c1fe5e ; trace sha256 8f17420214d2dd9df3cf7f44904b7cae24a849fc366b8e6cf16e4676c5e1dfdc
T=trace.npz; WHY="OOM at step 18 (index 17): the A2000 carried ~1.5 GiB of other resident tenants; the cut is set by micro-batch length, not routing; decided before any hit rate"
python bench/train-routing-trace/cross_step_bar.py --trace $T --expect-mb 8 --budget 768 --first-scored 5 --null-seed 1 --complete-steps 17 --truncation-reason "$WHY" --out bar.json
python bench/train-routing-trace/replay.py --trace $T --expect-mb 8 --complete-steps 17 --truncation-reason "$WHY" --fit-steps 5 --out replay.json
python bench/train-routing-trace/replay.py --trace $T --expect-mb 8 --complete-steps 17 --truncation-reason "$WHY" --fit-steps 5 --null-seed 1 --out null.json
for f in code dialogue math prose; do python bench/train-routing-trace/replay.py --decode <grouped-nf4-gemm 0677fddb>/bench/cold-engine/routing-trace/qwen3_$f.jsonl --fit-steps 128 --out decode-$f.json; done
