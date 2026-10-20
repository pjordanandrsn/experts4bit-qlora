### P149: the A2000 router pre-read's g16_routing band in the Predictions; the dry run's slice stays on the box

Before box Q+S is created, as Amendment 4 requires, the PREREG's Predictions now carry the USD 0 A2000 router pre-read
(`bench/p149/preread/preread.json`). S1's greedy W16 stream differs from S0's in all 16 rows, but replayed through
S0's router the two touch 7.8919 and 7.8927 distinct experts per layer-step, every expert inside one 16-row tile. That
predicts g16_routing 0.99991, with a band of [0.9979, 1.0019]. It is graded in the read and never rules. The driver's
fetch now leaves a dry run's one-layer cut (`slice/`, 3.2 GB) on the box, as `p149-dry-2` showed it should.
