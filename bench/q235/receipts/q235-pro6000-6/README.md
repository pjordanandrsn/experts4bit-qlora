# `q235-pro6000-6`: VOID at the repeat's time-left guard (rc 142), strike one

One RTX PRO 6000 Blackwell Workstation Edition on machine 46209 (Vast instance 56593481), e4b `616a67f1`. Cost $3.848. The
box phase completed. The fresh-process repeat then ran past its 705 s cap, and its time-left guard ended it (rc 142). That is
a box-started VOID: strike one. Amendment 15 then sized the repeat's cap on the box: max(ceil(1.5 × (load + 5 × step)), 60),
bounded by the time left less 300 s.

The launcher's receipt and ledger row are in the private receipt store. `SHA256SUMS` covers every file here.
